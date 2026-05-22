

# WORLD STATE, Weight Matrix





# from https://github.com/ml-jku/hopfield-layers/blob/master/hflayers/__init__.py












class HopfieldLayer(Module):
    """
    Wrapper class encapsulating a trainable but fixed stored pattern, pattern projection and "Hopfield" in
    one combined module to be used as a Hopfield-based pooling layer.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 pattern_size: Optional[int] = None,
                 num_heads: int = 1,
                 scaling: Optional[Union[float, Tensor]] = None,
                 update_steps_max: Optional[Union[int, Tensor]] = 0,
                 update_steps_eps: Union[float, Tensor] = 1e-4,
                 lookup_weights_as_separated: bool = False,
                 lookup_targets_as_trainable: bool = True,

                 normalize_stored_pattern: bool = True,
                 normalize_stored_pattern_affine: bool = True,
                 normalize_state_pattern: bool = True,
                 normalize_state_pattern_affine: bool = True,
                 normalize_pattern_projection: bool = True,
                 normalize_pattern_projection_affine: bool = True,
                 normalize_hopfield_space: bool = False,
                 normalize_hopfield_space_affine: bool = False,
                 stored_pattern_as_static: bool = False,
                 state_pattern_as_static: bool = False,
                 pattern_projection_as_static: bool = False,
                 pattern_projection_as_connected: bool = False,
                 stored_pattern_size: Optional[int] = None,
                 pattern_projection_size: Optional[int] = None,

                 batch_first: bool = True,
                 association_activation: Optional[str] = None,
                 dropout: float = 0.0,
                 input_bias: bool = True,
                 concat_bias_pattern: bool = False,
                 add_zero_association: bool = False,
                 disable_out_projection: bool = False,
                 quantity: int = 1,
                 trainable: bool = True
                 ):
        """
        Initialise a new instance of a Hopfield-based lookup layer.

        :param input_size: depth of the input (state pattern)
        :param hidden_size: depth of the association space
        :param output_size: depth of the output projection
        :param pattern_size: depth of patterns to be selected
        :param num_heads: amount of parallel association heads
        :param scaling: scaling of association heads, often represented as beta (one entry per head)
        :param update_steps_max: maximum count of association update steps (None equals to infinity)
        :param update_steps_eps: minimum difference threshold between two consecutive association update steps
        :param lookup_weights_as_separated: separate lookup weights from lookup target weights
        :param lookup_targets_as_trainable: employ trainable lookup target weights (used as pattern projection input)
        :param normalize_stored_pattern: apply normalization on stored patterns
        :param normalize_stored_pattern_affine: additionally enable affine normalization of stored patterns
        :param normalize_state_pattern: apply normalization on state patterns
        :param normalize_state_pattern_affine: additionally enable affine normalization of state patterns
        :param normalize_pattern_projection: apply normalization on the pattern projection
        :param normalize_pattern_projection_affine: additionally enable affine normalization of pattern projection
        :param normalize_hopfield_space: enable normalization of patterns in the Hopfield space
        :param normalize_hopfield_space_affine: additionally enable affine normalization of patterns in Hopfield space
        :param stored_pattern_as_static: interpret specified stored patterns as being static
        :param state_pattern_as_static: interpret specified state patterns as being static
        :param pattern_projection_as_static: interpret specified pattern projections as being static
        :param pattern_projection_as_connected: connect pattern projection with stored pattern
        :param stored_pattern_size: depth of input (stored pattern)
        :param pattern_projection_size: depth of input (pattern projection)
        :param batch_first: flag for specifying if the first dimension of data fed to "forward" reflects the batch size
        :param association_activation: additional activation to be applied on the result of the Hopfield association
        :param dropout: dropout probability applied on the association matrix
        :param input_bias: bias to be added to input (state and stored pattern as well as pattern projection)
        :param concat_bias_pattern: bias to be concatenated to stored pattern as well as pattern projection
        :param add_zero_association: add a new batch of zeros to stored pattern as well as pattern projection
        :param disable_out_projection: disable output projection
        :param quantity: amount of stored patterns
        :param trainable: stored pattern used for lookup is trainable
        """
        super(HopfieldLayer, self).__init__()
        self.hopfield = Hopfield(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size, pattern_size=pattern_size,
            num_heads=num_heads, scaling=scaling, update_steps_max=update_steps_max, update_steps_eps=update_steps_eps,
            normalize_stored_pattern=normalize_stored_pattern,
            normalize_stored_pattern_affine=normalize_stored_pattern_affine,
            normalize_state_pattern=normalize_state_pattern,
            normalize_state_pattern_affine=normalize_state_pattern_affine,
            normalize_pattern_projection=normalize_pattern_projection,
            normalize_pattern_projection_affine=normalize_pattern_projection_affine,
            normalize_hopfield_space=normalize_hopfield_space,
            normalize_hopfield_space_affine=normalize_hopfield_space_affine,
            stored_pattern_as_static=stored_pattern_as_static, state_pattern_as_static=state_pattern_as_static,
            pattern_projection_as_static=pattern_projection_as_static,
            pattern_projection_as_connected=pattern_projection_as_connected, stored_pattern_size=stored_pattern_size,
            pattern_projection_size=pattern_projection_size, batch_first=batch_first,
            association_activation=association_activation, dropout=dropout, input_bias=input_bias,
            concat_bias_pattern=concat_bias_pattern, add_zero_association=add_zero_association,
            disable_out_projection=disable_out_projection)
        self._quantity = quantity












        lookup_weight_size = self.hopfield.hidden_size if stored_pattern_as_static else self.hopfield.stored_pattern_dim
        
        
        # Mmeory matrix, or "world state" matrix, which is trainable and used as the stored pattern for the Hopfield lookup. The pattern projection can be either the same as the stored pattern or a separate trainable parameter
        
        self.lookup_weights = nn.Parameter(torch.empty(size=(*(
            (1, quantity) if batch_first else (quantity, 1)
        ), input_size if lookup_weight_size is None else lookup_weight_size)), requires_grad=trainable)



        if lookup_weights_as_separated:
            target_weight_size = self.lookup_weights.shape[
                2] if pattern_projection_size is None else pattern_projection_size
            self.target_weights = nn.Parameter(torch.empty(size=(*(
                (1, quantity) if batch_first else (quantity, 1)
            ), target_weight_size)), requires_grad=lookup_targets_as_trainable)
        else:
            self.register_parameter(name=r'target_weights', param=None)
        self.reset_parameters()













    def reset_parameters(self) -> None:
        """
        Reset lookup and lookup target weights, including underlying Hopfield association.

        :return: None
        """
        if hasattr(self.hopfield, r'reset_parameters'):
            self.hopfield.reset_parameters()

        # Explicitly initialise lookup and target weights.
        nn.init.normal_(self.lookup_weights, mean=0.0, std=0.02)
        if self.target_weights is not None:
            nn.init.normal_(self.target_weights, mean=0.0, std=0.02)

    def _prepare_input(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Prepare input for Hopfield association.

        :param input: data to be prepared
        :return: stored pattern, expanded state pattern as well as pattern projection
        """
        batch_size = input.shape[0 if self.batch_first else 1]
        stored_pattern = self.lookup_weights.expand(size=(*(
            (batch_size, self.quantity) if self.batch_first else (self.quantity, batch_size)
        ), self.lookup_weights.shape[2]))
        if self.target_weights is None:
            pattern_projection = stored_pattern
        else:
            pattern_projection = self.target_weights.expand(size=(*(
                (batch_size, self.quantity) if self.batch_first else (self.quantity, batch_size)
            ), self.target_weights.shape[2]))

        return stored_pattern, input, pattern_projection

    def forward(self, input: Tensor, stored_pattern_padding_mask: Optional[Tensor] = None,
                association_mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute Hopfield-based lookup on specified data.

        :param input: data to used in lookup
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: result of Hopfield-based lookup on input data
        """
        return self.hopfield(
            input=self._prepare_input(input=input),
            stored_pattern_padding_mask=stored_pattern_padding_mask,
            association_mask=association_mask)

    def get_association_matrix(self, input: Tensor, stored_pattern_padding_mask: Optional[Tensor] = None,
                               association_mask: Optional[Tensor] = None) -> Tensor:
        """
        Fetch Hopfield association matrix used for lookup gathered by passing through the specified data.

        :param input: data to be passed through the Hopfield association
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: association matrix as computed by the Hopfield core module
        """
        with torch.no_grad():
            return self.hopfield.get_association_matrix(
                input=self._prepare_input(input=input),
                stored_pattern_padding_mask=stored_pattern_padding_mask,
                association_mask=association_mask)

    def get_projected_pattern_matrix(self, input: Union[Tensor, Tuple[Tensor, Tensor]],
                                     stored_pattern_padding_mask: Optional[Tensor] = None,
                                     association_mask: Optional[Tensor] = None) -> Tensor:
        """
        Fetch Hopfield projected pattern matrix gathered by passing through the specified data.

        :param input: data to be passed through the Hopfield association
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: pattern projection matrix as computed by the Hopfield core module
        """
        with torch.no_grad():
            return self.hopfield.get_projected_pattern_matrix(
                input=self._prepare_input(input=input),
                stored_pattern_padding_mask=stored_pattern_padding_mask,
                association_mask=association_mask)






















# -------------------------------------------------------------------





#  External Memory Bank (Hopfield + Memory Network)
# Separate the memory storage from the Hopfield layer entirely:




class ExternalMemoryHopfield(nn.Module):
    def __init__(self, feature_dim, memory_slots=1000):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_slots = memory_slots
        
        # External memory buffer (not a parameter - updated manually)
        self.register_buffer('memory_bank', torch.zeros(memory_slots, feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('usage_count', torch.zeros(memory_slots))
        
        # Hopfield layer for reading
        self.reader = HopfieldLayer(
            input_size=feature_dim,
            quantity=memory_slots,  # Will use our external memory instead
            stored_pattern_as_static=True
        )
        
        # Compressor network (optional): compress output before storing
        self.compressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
    def write(self, output_repr):
        """Write output representation to external memory"""
        # Compress representation
        compressed = self.compressor(output_repr.detach()).mean(dim=0)  # Average over batch
        
        # Circular buffer write
        ptr = self.memory_ptr.item()
        self.memory_bank[ptr] = compressed
        self.usage_count[ptr] = 1
        
        # Update pointer
        self.memory_ptr[0] = (ptr + 1) % self.memory_slots
        
    def read(self, query):
        """Read from external memory using Hopfield"""
        # Override HopfieldLayer's internal lookup_weights with our memory_bank
        # This requires modifying HopfieldLayer or using Hopfield directly
        batch_size = query.shape[0]
        
        # Expand memory: (memory_slots, feature_dim) -> (batch, memory_slots, feature_dim)
        memory_expanded = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Use HopfieldCore directly for more control
        return self.reader.hopfield((memory_expanded, query, memory_expanded))
    
    def forward(self, x, store_output=False):
        output = self.read(x)
        
        if store_output:
            self.write(output)
            
        return output
    
    def get_memory_stats(self):
        """Check memory utilization"""
        return {
            'filled_slots': (self.memory_bank.abs().sum(dim=1) > 0).sum().item(),
            'total_slots': self.memory_slots,
            'usage_distribution': self.usage_count.clone()
        }