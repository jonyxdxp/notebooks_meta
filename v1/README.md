

x ---> "Metalearner" ("sparse-sr-fwp" and "cms") ---generates (by the "Optimizer", in "test_env", and Verifying the Self-Modification by "PoK")---> "cog. arch." ---stored---> "DAG (Checkpoints)" ---> y  


y ---stored (in Inference)---> "World State" ---varied by---> Context AGENT (by Energy Minimization) ---> y (Minimized/denoised)




(this is all we need to VALIDATE now)





_____________________________________



(we need some preprosesing step before giving "x" to the "metarlearner0")




* Metalearner: comprised by the "sparse-sr-fwp" and "cms" modules;

will have two parts: encoder-like (to train the "encoder" and "dm" of the Cog. Arch.) and decoder-like (to train the "policy" and "decoder")


- Sparse-sr-fwp: similar implementation to "self-referential titans"; this is composed of the "Slow Weights", the "Optimizer" and the "Fast Weights" (generated Cog. Arch.)

--- the "Optimizer" may be some "gradient descent with momentum" variant, and we need to have a way to store its "State" so we can "optimize/train" the Cog. Arch. in one-shot at inference-time, so it may seems like its "generated"...... and will be "generated" in the "test_env" wich is a minimal emulation of the "mvm", wich has to have all the necesary components to implement the "execution part" of the METANET......

--- the "Cog. Arch." will be a standart architecture to develop AGENTS in the Net, this will consist in modules like "encoder - dm - policy - memory - decoder".... (the "memory" module will be the only one that will not be "fast/generated" as it will be stored in a subestructure of the "Checkpoint", and will be used to retrain the "Policy", and "dm"? when the Observation after some Action doesn't yield where expected)


- cms: this is just a replacement for the logic of the mlp/ffs network of transformers-like arquitectures


----- training the second part of the Metalearner: we need to condition the the Metalearner in the encodings made by the first part, it will be its main supervised input signal and the output supervision will be some decoded version of the prediction;

now, we need first to see how to train the "Policy": it will need to look for actions in some existent Agent of te Net, or to look in the algorithmic information conveyed by the "dm".... those will be its only source of action sigals to look/search for

one of the most interesting part of the "dm" its that it will make prediction over its own input, like a bert model, but conditioned on an action signal provided by the "Policy", it will predict in the future.....



* PoK: before the "self-referent update" by the "sr-fwp", we will need to apply some "search algorithm" to look in the DAG's Checkpoints for "Optimizers" whose stored "State"? generated the respective version of "Cog. Arch"..... another method could be to make "recurrent" the "Optimizer" so we would have diferent outputs for the same input at diferent recurrent-steps (may be something like dynamic programming)???



* DAG: minimal implementation of a DAG-based blockchain, withh "Checkpoints" as nodes, whose structure will define information like: Agent metadata, Cog. Arch. (with Short-Term Memory storage), Optimizer State(?), etc..... the "PoK" will search over the "Optimizers states" with the pupose to minimize the "entropy/surprise term" of the Task Loss



* Inference by Minimizing the Energy of States with respect to Latent Variables: this is one of the main characteristics of the METANET, and consist of the optimization process that occurs during inference in the "dm";

first, some input pass throught the "dm" and its "Energy" (deduced in the training stage) is computed, then we store those intermediated states obtained in the diferent levels of abstraction (determined by the "self-referential update" process to minimize the entropy/surprise term of the loss) in the "World State" structure, wich will be a "Matrix valued memory" (different than that of the "st-memory" wich will be a NN), whose entries will be the "intermediates states" with its Energy 

then, by aplying the "denoising/minimizing" process to the intermediate states with respect to the "context agent", we obtain the output
































