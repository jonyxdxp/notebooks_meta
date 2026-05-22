

# mamba2 byte encoder network as in "hnet"





# from https://github.com/main-horse/hnet-impl/blob/master/src/hnet_impl/xf.py



from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated



class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()
        assert d_model * expand // headdim % 8 == 0, (
            "https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940"
        )
        assert d_conv <= 4, "causal-conv1d only supports d_conv <= 4"

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size

        self.d_inner = self.expand * self.d_model
        self.d_ssm = self.d_inner  # full dim SSM
        assert (self.d_ssm % self.headdim) == 0, (
            "expand*d_model must be divisible by headdim"
        )
        self.nheads = self.d_ssm // self.headdim

        # NOTE: to reduce complexity, I hardcode the following behaviors:
        self.ngroups = 1
        self.activation = "silu"
        self.norm_before_gate = False

        # projections
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=True,
        )

        # force mamba params to init, even if meta device init is used
        with torch.device("cuda"):
            p = self.construct_ssm_params(self.nheads)
            self.A_log = nn.Parameter(p["A_log"])
            self.dt_bias = nn.Parameter(p["dt_bias"])
            self.D = nn.Parameter(p["D"])

        # normalisation & output
        self.norm = RMSNormGated(
            self.d_ssm, eps=1e-5, norm_before_gate=False, group_size=self.d_ssm
        )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    # NOTE: init choices here are taken from mamba2 defaults.
    @staticmethod
    def construct_ssm_params(
        nheads: int,
        *,
        A_init_range=(1.0, 16.0),
        dt_min=1e-3,
        dt_max=1e-1,
        dt_init_floor=1e-4,
    ):
        rand = torch.rand(nheads, dtype=torch.float32)
        exponent = rand * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        dt = torch.exp(exponent).clamp_(min=dt_init_floor)

        return dict(
            dt_bias=dt + torch.log(-torch.expm1(-dt)),
            A_log=torch.log(
                torch.empty(nheads, dtype=torch.float32).uniform_(*A_init_range)
            ),
            D=torch.ones(nheads),
        )

    def forward(self, u: TT, seq_idx: TT) -> TT:
        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log.float())
        out = mamba_split_conv1d_scan_combined(
            zxbcdt,
            self.conv1d.weight.squeeze(-2),
            self.conv1d.bias,
            self.dt_bias.type_as(u),
            A,
            D=self.D.type_as(u),
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.eps,
            headdim=self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
        )
        # NOTE: I move out_proj outside of mamba2 kernel, in case fp8 is desired.
        return self.out_proj(out)
    

















# from https://github.com/main-horse/hnet-impl/blob/master/src/hnet_impl/modeling_hnet.py





from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from .torchisms import torch, TT, nn, F, nested, NJT, summon_full_params
from .conceptual import BlockBoundaryMixin, get_seq_idx
from .config_hnet import HNetConfig
from .xf import Isotropic
from .lin import Lin, HighPrecLinear, LMHead


### ################
### H-Net submodules
### ################


@dataclass(frozen=True)
class HNetExtra:
    b: TT  # (B,j1) boolean label for whether byte was selected
    loss_ratio: TT  # scalar tensor -- routing loss for this block
    compress_ratio: float  # scalar float -- compression ratio


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_func(x):
    return STE.apply(x)


class QProjPadded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat: TT, w: TT, k_flat: TT, cu: TT):
        slen = x_flat.shape[0]
        # compute x@w.T, but padded left by 1seqlen
        q_padded = torch.empty(
            slen + 1, *x_flat.shape[1:], dtype=x_flat.dtype, device=x_flat.device
        )
        torch.mm(x_flat, w.T.type_as(x_flat), out=q_padded[1:])
        ctx.save_for_backward(x_flat, w, cu)
        return q_padded.index_copy_(0, cu[:-1], -k_flat[cu[:-1]])[:slen]

    @staticmethod
    def backward(ctx, dq_flat: TT):
        x_flat, w, cu = ctx.saved_tensors
        zero_grad = torch.zeros(
            cu.shape[0] - 1,
            dq_flat.shape[-1],
            device=dq_flat.device,
            dtype=dq_flat.dtype,
        )
        dq_flat = dq_flat.index_copy(0, cu[:-1], zero_grad)

        dx_flat = torch.zeros_like(x_flat)
        torch.mm(dq_flat[1:], w.type_as(dq_flat), out=dx_flat[:-1])
        dw = dq_flat[1:].mT @ x_flat[:-1]

        return dx_flat, dw, None, None


# NOTE: it's possible to fuse q/k/res proj into a single gemm kernel, but only iff they are of equal precision.
class RoutingModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.q_proj_layer = Lin(d, d)
        self.k_proj_layer = Lin(d, d)
        # https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py#L49
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d))
            self.k_proj_layer.weight.copy_(torch.eye(d))

    def forward(self, r_flat: TT, r_cu: TT):
        k_flat = self.k_proj_layer(r_flat)
        q_flat = QProjPadded.apply(r_flat, self.q_proj_layer.weight, k_flat, r_cu)
        cos_sim = F.cosine_similarity(q_flat, k_flat, dim=-1)
        p_flat = (0.5 - cos_sim / 2).clamp(0.0, 1.0)
        b_flat = p_flat >= 0.5
        p_select_cu = F.pad(b_flat.cumsum(0), (1, 0))[r_cu]
        return p_flat, b_flat, p_select_cu


class DeChunkLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # for EMA scan kernel.
        self.block_size = 256
        self.headdim = 32
        self.nheads, _r = divmod(d, self.headdim)
        assert _r == 0
        A = -torch.ones(self.nheads, device="cuda", dtype=torch.float32)
        self.register_buffer("A", A, persistent=False)

    @staticmethod
    def forward_flat(
        h_flat: TT,
        b_flat: TT,
        p_selected_flat: TT,
        h_seq_idx: TT,
        *,
        eps=1e-4,
        nheads: int,
        headdim: int,
        block_size: int,
        A: TT,
    ):
        p = p_selected_flat.float().clamp(eps, 1 - eps)

        dt = -torch.log1p(-p.float())[..., None]
        h = (h_flat.float() / dt).type_as(h_flat)
        c = torch.ones_like(p := p.type_as(h)[None, :, None, None])

        z_bar_flat = mamba_chunk_scan_combined(
            h.view(1, -1, nheads, headdim),
            dt.expand(-1, nheads).to(h.dtype)[None],
            A,
            p,
            c,
            chunk_size=block_size,
            seq_idx=h_seq_idx,
        )[0].view(-1, h.shape[-1])

        inner2outer_idx = b_flat.cumsum(0) - 1
        return z_bar_flat.index_select(0, inner2outer_idx)

    def forward(
        self, h_flat: TT, b_flat: TT, p_selected_flat: TT, h_seq_idx: TT, *, eps=1e-4
    ):
        return self.forward_flat(
            h_flat,
            b_flat,
            p_selected_flat,
            h_seq_idx,
            eps=eps,
            nheads=self.nheads,
            headdim=self.headdim,
            block_size=self.block_size,
            A=self.A,
        )