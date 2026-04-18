
# local Action encoder for the high-level Agent





# from https://github.com/facebookresearch/vjepa2/blob/main/src/models/ac_predictor.py



# Map input to predictor dimension
    self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
    self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
    self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
    self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)



        s = self.state_encoder(states).unsqueeze(2)
        a = self.action_encoder(actions).unsqueeze(2)
        x = x.view(B, T, self.grid_height * self.grid_width, D)  # [B, T, H*W, D]
        if self.use_extrinsics:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)  # [B, T*(H*W+3), D]
        else:
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)  # [B, T*(H*W+2), D]







# from https://github.com/facebookresearch/vjepa2/blob/main/src/models/utils/modules.py


 if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)