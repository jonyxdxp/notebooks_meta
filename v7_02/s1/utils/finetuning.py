# finetuning logic for the Learned Optimizer (Lopt) algorithm








# from https://github.com/mikehuisman/revisiting-learned-optimizers/blob/main/algorithms/finetuning.py








from .algorithm import Algorithm
from .modules.utils import eval_model, get_batch, new_weights, update,\
                   put_on_device, deploy_on_task

class FineTuning(Algorithm):
    """Transfer-learning model based on pre-training and fine-tuning
    
    Model that pre-trains on batches from all meta-training data,
    and finetunes only the last layer when presented with tasks at
    meta-validation/meta-test time.
    
    ...

    Attributes
    ----------
    baselearner_fn : constructor function
        Constructor function for the base-learner
    baselearner_args : dict
        Dictionary of keyword arguments for the base-learner
    opt_fn : constructor function
        Constructor function for the optimizer to use
    T : int
        Number of update steps to parameters per task
    train_batch_size : int
        Indicating the size of minibatches that are sampled from meta-train tasks
    test_batch_size : int
        Size of batches to sample from meta-[val/test] tasks
    lr : float
        Learning rate for the optimizer
    cpe : int
        Checkpoints per episode (# times we recompute new best weights)
    dev : str
        Device identifier
    trainable : boolean
        Whether it can train on meta-trrain tasks
    frozen : boolean
        Flag whether the weights of all except the final layer has been frozen
        
    Methods
    -------
    _freeze_layers()
        Freeze all hidden layers
        
    train(train_x, train_y, **kwargs)
        Train on a given batch of data
        
    evaluate(train_x, train_y, test_x, test_y)
        Evaluate the performance on the given task
        
    dump_state()
        Dumps the model state
        
    load_state(state)
        Loads the given model state    
    """
    
    def __init__(self, cpe, **kwargs):
        """
        Call parent constructor function to inherit and set attributes
        Create a model that will be used throughout. Set frozen state to be false.
        
        Parameters
        ----------
        cpe : int
            Number of times the best weights should be reconsidered in an episode
        """
        
        super().__init__(**kwargs)
        self.cpe = cpe
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.baselearner.train()
        self.val_learner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.val_learner.load_state_dict(self.baselearner.state_dict())
        self.val_learner.train()

        #print(self.baselearner.model.out.prototypes.device); import sys; sys.exit()
        self.optimizer = self.opt_fn(self.baselearner.parameters(), lr=self.lr)
        self.episodic = False
                
    def train(self, train_x, train_y, **kwargs):
        """Train on batch of data
        
        Train for a single step on the support set data
        
        Parameters
        ----------
        train_x : torch.Tensor
            Tensor of inputs
        train_y : torch.Tensor
            Tensor of ground-truth outputs corresponding to the inputs
        **kwargs : dict
            Ignore other parameters
        """
        
        self.baselearner.train()
        self.val_learner.train()
        #print("Bias nodes in output layer:", self.baselearner.model.out.bias.size())
        train_x, train_y = put_on_device(self.dev, [train_x, train_y])
        update(self.baselearner, self.optimizer, train_x, train_y)

        
    def evaluate(self, train_x, train_y, test_x, test_y):
        """Evaluate the model on a task
        
        Fine-tune the last layer on the support set of the task and
        evaluate on the query set.
        
        Parameters
        ----------
        train_x : torch.Tensor
            Tensor of inputs
        train_y : torch.Tensor
            Tensor of ground-truth outputs corresponding to the inputs
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Ground-truth outputs of the query set
        
        Returns
        ----------
        test_loss
            Loss (float) on the query set
        """
        
        # CHeck if the mode has changed (val_learner.training=True during training). 
        # If so, set weights of validation learner to those of base-learner
        change = self.val_learner.training != False
        if change:
            self.val_learner.load_state_dict(self.baselearner.state_dict())
            
        # Check if mode is changed
        self.val_learner.eval()
        
        # If hidden layers not frozen yet, make em cold!
        self.val_learner.freeze_layers()
        
        val_optimizer = self.opt_fn(self.val_learner.parameters(), self.lr)
        
        # Put on the right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev, 
                                            [train_x, train_y, 
                                             test_x, test_y])

        # Train on support set and get loss on query set
        test_loss = deploy_on_task(
                        model=self.val_learner, 
                        optimizer=val_optimizer,
                        train_x=train_x, 
                        train_y=train_y, 
                        test_x=test_x, 
                        test_y=test_y, 
                        T=self.T, 
                        test_batch_size=self.test_batch_size,
                        cpe=self.cpe,
                        init_score=self.init_score,
                        operator=self.operator
        )
        return test_loss
        
    def dump_state(self):
        """Return the state of the model
        
        Returns
        ----------
        state
            State dictionary of the base-learner model
        """
        return self.baselearner.state_dict()
    
    def load_state(self, state):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : dict
            Base-learner parameters
        """
        
        # Load state is only called to load task-model architectures,
        # so call eval mode (because number of classes in task model differs
        # from that of the non-task model
        self.baselearner.eval()
        self.baselearner.load_state_dict(state) 


















# ----------------------------------------------



# from https://github.com/galilai-group/llm-jepa/blob/main/finetune.py








def setup_model_and_tokenizer(model_name, use_lora=True, lora_rank=16, pretrain=False, debug=0, seed=None):
    """Setup model and tokenizer with optional LoRA"""
    
    # Load tokenizer
    if "apple/OpenELM" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert tokenizer.chat_template is not None, f"{model_name} does not have chat template."
    
    # use_llama_3_2_chat_template(tokenizer)
    
    # Add special tokens if not present
    if "microsoft/phi" in model_name:
        tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
        if torch.cuda.current_device() == 0:
            print("Added <|startoftext|> token")

    special_tokens = ["<|predictor_1|>", "<|predictor_2|>", "<|predictor_3|>", "<|predictor_4|>", "<|predictor_5|>",
                      "<|predictor_6|>", "<|predictor_7|>", "<|predictor_8|>", "<|predictor_9|>", "<|predictor_10|>",
                      "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|perception|>"]
    new_tokens = [token for token in special_tokens if token not in tokenizer.vocab]
    
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if torch.cuda.current_device() == 0:
            print(f"Added {len(new_tokens)} new special tokens")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with better device mapping for multi-GPU
    device_map = None
    if torch.cuda.is_available():
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if world_size == 1:
            device_map = "auto"
        else:
            # For multi-GPU with torchrun, don't use device_map
            device_map = None
    
    if pretrain:
        if seed is not None:
            torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        model.to(device)
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)
        for b in model.buffers():
            torch.distributed.broadcast(b.data, src=0)
        if debug == 6:
            for name, param in model.named_parameters():
                print(f"Parameter name: {name}, Shape: {param.shape}")
                print(param)
                exit(0)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            # Add these for better multi-GPU stability
            low_cpu_mem_usage=True,
            use_cache=False,  # Disable KV cache for training
        )

    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
    
    # Resize embeddings if we added new tokens
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
    
    # Setup LoRA if requested
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        if torch.cuda.current_device() == 0:
            model.print_trainable_parameters()
    
    return model, tokenizer