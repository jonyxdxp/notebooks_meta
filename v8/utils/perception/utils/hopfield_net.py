

# from https://github.com/ml-jku/mhn-react/blob/main/mhnreact/model.py
















class MHN(nn.Module):
    """
    MHN - modern Hopfield Network -- for Template relevance prediction
    """
    def __init__(self, config=None, layer2weight=0.05, use_template_encoder=True):
        super().__init__()
        if config:
            self.config = config
        else:
            self.config = ModelConfig()
        self.beta = self.config.hopf_beta
        # hopf_num_heads
        self.mol_encoder = FPMolEncoder(self.config)
        if use_template_encoder:
            self.template_encoder = TemplateEncoder(self.config)

        self.W_v = None
        self.layer2weight = layer2weight

        # more MHN layers -- added recursively
        if hasattr(self.config, 'hopf_n_layers'):
            di = self.config.__dict__
            di['hopf_n_layers'] -= 1
            if di['hopf_n_layers']>0:
                conf_wo_hopf_nlayers = ModelConfig(**di)
                self.layer = MHN(conf_wo_hopf_nlayers)
                if di['hopf_n_layers']!=0:
                    self.W_v = nn.Linear(self.config.hopf_asso_dim, self.config.hopf_input_size)
                    torch.nn.init.kaiming_normal_(self.W_v.weight, mode='fan_in', nonlinearity='linear') # eqiv to LeCun init

        self.softmax = torch.nn.Softmax(dim=1)

        self.lossfunction = nn.CrossEntropyLoss(reduction='none')#, weight=class_weights)
        self.pretrain_lossfunction = nn.BCEWithLogitsLoss(reduction='none')#, weight=class_weights)

        self.lr = self.config.lr

        if self.config.hopf_association_activation is None or (self.config.hopf_association_activation.lower()=='none'):
            self.af = lambda k: k
        else:
            self.af = getattr(nn, self.config.hopf_association_activation)()

        self.pooling_operation_head = getattr(torch, self.config.pooling_operation_head)

        self.X = None # templates projected to Hopfield Layer

        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.parameters(), lr=self.lr)
        self.steps = 0
        self.hist = defaultdict(list)
        self.to(self.config.device)

    def set_templates(self, template_list, which='rdk', fp_size=None, radius=2, learnable=False, njobs=1, only_templates_in_batch=False):
        self.template_list = template_list.copy()
        if fp_size is None:
            fp_size = self.config.fp_size
        if len(template_list)>=100000:
            import math
            print('batch-wise template_calculation')
            bs = 30000
            final_temp_emb = torch.zeros((len(template_list), fp_size)).float().to(self.config.device)
            for b in range(math.ceil(len(template_list)//bs)+1):
                self.template_list = template_list[bs*b:min(bs*(b+1), len(template_list))]
                templ_emb = self.update_template_embedding(which=which, fp_size=fp_size, radius=radius, learnable=learnable, njobs=njobs, only_templates_in_batch=only_templates_in_batch)
                final_temp_emb[bs*b:min(bs*(b+1), len(template_list))] = torch.from_numpy(templ_emb)
            self.templates = final_temp_emb
        else:
            self.update_template_embedding(which=which, fp_size=fp_size, radius=radius, learnable=learnable, njobs=njobs, only_templates_in_batch=only_templates_in_batch)
        
        self.set_templates_recursively()

    def set_templates_recursively(self):
        if 'hopf_n_layers' in self.config.__dict__.keys():
            if self.config.hopf_n_layers >0:
                self.layer.templates = self.templates
                self.layer.set_templates_recursively()

    def update_template_embedding(self,fp_size=2048, radius=4, which='rdk', learnable=False, njobs=1, only_templates_in_batch=False, template_list=None, verbose=True):
        if verbose: print('updating template-embedding; (just computing the template-fingerprint and using that)')
        bs = self.config.batch_size

        template_list = self.template_list if template_list is None else template_list

        split_template_list = [str(t).split('>')[0].split('.') for t in template_list]
        templates_np = convert_smiles_to_fp(split_template_list, is_smarts=True, fp_size=fp_size, radius=radius, which=which, njobs=njobs)

        split_template_list = [str(t).split('>')[-1].split('.') for t in template_list]
        reactants_np = convert_smiles_to_fp(split_template_list, is_smarts=True, fp_size=fp_size, radius=radius, which=which, njobs=njobs)

        template_representation = templates_np-(reactants_np*0.5)
        if learnable:
            self.templates = torch.nn.Parameter(torch.from_numpy(template_representation).float(), requires_grad=True).to(self.config.device)
            self.register_parameter(name='templates', param=self.templates)
        else:
            if only_templates_in_batch:
                self.templates_np = template_representation
            else:
                self.templates = torch.from_numpy(template_representation).float().to(self.config.device)
                
        return template_representation


    def np_fp_to_tensor(self, np_fp):
        return torch.from_numpy(np_fp.astype(np.float64)).to(self.config.device).float()

    def masked_loss_fun(self, loss_fun, h_out, ys_batch):
        if loss_fun == self.BCEWithLogitsLoss:
            mask = (ys_batch != -1).float()
            ys_batch = ys_batch.float()
        else:
            mask = (ys_batch.long() != -1).long()
        mask_sum = int(mask.sum().cpu().numpy())
        if mask_sum == 0:
            return 0

        ys_batch = ys_batch * mask

        loss = (loss_fun(h_out, ys_batch * mask) * mask.float()).sum() / mask_sum  # only mean from non -1
        return loss

    def compute_losses(self, out, ys_batch, head_loss_weight=None):

        if len(ys_batch.shape)==2:
            if ys_batch.shape[1]==self.config.num_templates: # it is in pretraining_mode
                loss = self.pretrain_lossfunction(out, ys_batch.float()).mean()
            else:
                # legacy from policyNN
                loss = self.lossfunction(out, ys_batch[:, 2]).mean()  # WARNING: HEAD4 Reaction Template is ys[:,2]
        else:
            loss = self.lossfunction(out, ys_batch).mean() 
        return loss

    def forward_smiles(self, list_of_smiles, templates=None):
        state_tensor = self.mol_encoder.convert_smiles_to_tensor(list_of_smiles)
        return self.forward(state_tensor, templates=templates)

    def encode_templates(self, list_of_smarts, batch_size=32, njobs=1):
        """encodes a list of templates to a numpy array"""
        x = np.empty((len(list_of_smarts), self.config.hopf_asso_dim))
        for b in range(0, len(list_of_smarts), batch_size):
            # compute template fingerprints
            template_list = list_of_smarts[b:min(b+batch_size, len(list_of_smarts))]
            templ_imp_emb = self.update_template_embedding(which=self.config.template_fp_type, 
                                fp_size=self.config.fp_size, radius=self.config.fp_radius, learnable=False, njobs=njobs, 
                                only_templates_in_batch=True, template_list=template_list, verbose=False)
            templ_imp_emb = torch.from_numpy(templ_imp_emb).float().to(self.config.device)
            bx = self.template_encoder(templ_imp_emb).detach().cpu().numpy()

            x[b:min(b+batch_size, len(list_of_smarts))] = bx
        return x

    def encode_smiles(self, list_of_smiles, batch_size=32, njobs=1):
        """encodes a list of smiles to a numpy array"""
        x = np.empty((len(list_of_smiles), self.config.hopf_asso_dim))
        for b in range(0, len(list_of_smiles), batch_size):
            # compute template fingerprints
            smiles_list = list_of_smiles[b:min(b+batch_size, len(list_of_smiles))]
            mol_imp_emb = self.mol_encoder.convert_smiles_to_tensor(smiles_list)
            bx = self.mol_encoder(mol_imp_emb).detach().cpu().numpy()

            x[b:min(b+batch_size, len(list_of_smiles))] = bx
        return x

    def forward(self, m, templates=None):
        """
        m: molecule in the form batch x fingerprint
        templates: None or newly given templates if not instanciated
        returns logits ranking the templates for each molecule
        """

        bs = m.shape[0] #batch_size

        if (templates is None) and (self.X is None) and (self.templates is None):
            raise Exception('Either pass in templates, or init templates by runnting clf.set_templates')
        n_temp = len(templates) if templates is not None else len(self.templates)
        if self.training or (templates is not None) or (self.X is None):
            templates = templates if templates is not None else self.templates
            X = self.template_encoder(templates)
        else:
            X = self.X # precomputed from last forward run
        
        Xi = self.mol_encoder(m)

        Xi = Xi.view(bs, self.config.hopf_num_heads, self.config.hopf_asso_dim) # [bs, H, A]
        X = X.view(1, n_temp, self.config.hopf_asso_dim, self.config.hopf_num_heads) #[1, T, A, H]

        XXi = torch.tensordot(Xi, X, dims=[(2,1), (2,0)]) # AxA -> [bs, T, H]
        
        # pooling over heads
        if self.config.hopf_num_heads<=1:
            #QKt_pooled = QKt
            XXi = XXi[:,:,0] #torch.squeeze(QKt, dim=2)
        else:
            XXi = self.pooling_operation_head(XXi, dim=2) # default is max pooling over H [bs, T]
            if (self.config.pooling_operation_head =='max') or (self.config.pooling_operation_head =='min'):
                XXi = XXi[0] #max and min also return the indices =S

        out = self.beta*XXi # [bs, T, H] # softmax over dim=1 #pooling_operation_head
        
        self.xinew = self.softmax(out)@X.view(n_temp, self.config.hopf_asso_dim) # [bs,T]@[T,emb] -> [bs,emb]

        if self.W_v:
            # call layers recursive
            hopfout = self.W_v(self.xinew) # [bs,emb]@[emb,hopf_inp]  --> [bs, hopf_inp]
            # TODO check if using x_pooled or if not going through mol_encoder again
            hopfout = hopfout + m # skip-connection
            # give it to the next layer
            out2 = self.layer.forward(hopfout) #templates=self.W_v(self.K)
            out = out*(1-self.layer2weight)+out2*self.layer2weight

        return out

    def train_from_np(self, Xs, targets, ys, is_smiles=False, epochs=2, lr=0.001, bs=32,
                      permute_batches=False, shuffle=True, optimizer=None,
                      use_dataloader=True, verbose=False,
                      wandb=None, scheduler=None, only_templates_in_batch=False):
        """
        Xs in the form sample x states
        targets
        ys in the form sample x [y_h1, y_h2, y_h3, y_h4]
        """
        self.train()
        if optimizer is None:
            try:
                self.optimizer = getattr(torch.optim, self.config.optimizer)(self.parameters(), lr=self.lr if lr is None else lr)
            except AttributeError as err:
                log.error(f"Can't find optimizer {config.optimizer} in torch.optim")
                raise err
            optimizer = self.optimizer

        dataset = ChemRXNDataset(Xs, targets, ys, is_smiles=is_smiles,
                                 fp_size=self.config.fp_size, fingerprint_type=self.config.fingerprint_type)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, sampler=None,
                   batch_sampler=None, num_workers=0, collate_fn=None,
                   pin_memory=False, drop_last=False, timeout=0,
                   worker_init_fn=None)

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            running_loss_dict = defaultdict(int)
            batch_order = range(0, len(Xs), bs)
            if permute_batches:
                batch_order = np.random.permutation(batch_order)

            for step, s in tqdm(enumerate(dataloader),mininterval=2):
                batch = [b.to(self.config.device, non_blocking=True) for b in s]
                Xs_batch, target_batch, ys_batch = batch

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                out = self.forward(Xs_batch)
                total_loss = self.compute_losses(out, ys_batch)

                loss_dict = {'CE_loss': total_loss}

                total_loss.backward()

                optimizer.step()
                if scheduler:
                    scheduler.step()
                self.steps += 1

                # print statistics
                for k in loss_dict:
                    running_loss_dict[k] += loss_dict[k].item()
                try:
                    running_loss += total_loss.item()
                except:
                    running_loss += 0

                rs = min(100,len(Xs)//bs) # reporting/logging steps
                if step % rs == (rs-1):  # print every 2000 mini-batches
                    if verbose: print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, step + 1, running_loss / rs))
                    self.hist['step'].append(self.steps)
                    self.hist['loss'].append(running_loss/rs)
                    self.hist['trianing_running_loss'].append(running_loss/rs)

                    [self.hist[k].append(running_loss_dict[k]/rs) for k in running_loss_dict]

                    if wandb:
                        wandb.log({'trianing_running_loss': running_loss / rs})

                    running_loss = 0.0
                    running_loss_dict = defaultdict(int)

        if verbose: print('Finished Training')
        return optimizer

    def evaluate(self, Xs, targets, ys, split='test', is_smiles=False, bs = 32, shuffle=False, wandb=None, only_loss=False):
        self.eval()
        y_preds = np.zeros( (ys.shape[0], self.config.num_templates), dtype=np.float16)
        
        loss_metrics = defaultdict(int)
        new_hist = defaultdict(float)
        with torch.no_grad():
            dataset = ChemRXNDataset(Xs, targets, ys, is_smiles=is_smiles,
                                     fp_size=self.config.fp_size, fingerprint_type=self.config.fingerprint_type)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, sampler=None,
                       batch_sampler=None, num_workers=0, collate_fn=None,
                       pin_memory=False, drop_last=False, timeout=0,
                       worker_init_fn=None)

            #for step, s in eoutputs = self.forward(batch[0], batchnumerate(range(0, len(Xs), bs)):
            for step, batch in enumerate(dataloader):#
                batch = [b.to(self.config.device, non_blocking=True) for b in batch]
                ys_batch = batch[2]
                
                if hasattr(self, 'templates_np'):
                    outputs = []
                    for ii in range(10):
                        tlen = len(self.templates_np)
                        i_tlen = tlen//10
                        templates = torch.from_numpy(self.templates_np[(i_tlen*ii):min(i_tlen*(ii+1), tlen)]).float().to(self.config.device)
                        outputs.append( self.forward(batch[0], templates = templates ) )
                    outputs = torch.cat(outputs, dim=0)
                        
                else:
                    outputs = self.forward(batch[0]) 

                loss = self.compute_losses(outputs, ys_batch, None)

                # not quite right because in every batch there might be different number of valid samples
                weight = 1/len(batch[0])#len(Xs[s:min(s + bs, len(Xs))]) / len(Xs)

                loss_metrics['loss'] += (loss.item())

                if len(ys.shape)>1:
                    outputs = self.softmax(outputs) if not (ys.shape[1]==self.config.num_templates) else torch.sigmoid(outputs)
                else:
                    outputs = self.softmax(outputs)

                outputs_np = [None if o is None else o.to('cpu').numpy().astype(np.float16) for o in outputs]

                if not only_loss:
                    ks = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
                    topkacc, mrocc = top_k_accuracy(ys_batch, outputs, k=ks, ret_arocc=True, ret_mrocc=False)
                    # mrocc -- median rank of correct choice
                    for k, tkacc in zip(ks, topkacc):
                        #iterative average update
                        new_hist[f't{k}_acc_{split}'] += (tkacc-new_hist[f't{k}_acc_{split}']) / (step+1)
                        # todo weight by batch-size
                    new_hist[f'meanrank_{split}'] = mrocc

                y_preds[step*bs : min((step+1)*bs,len(y_preds))] = outputs_np


        new_hist[f'steps_{split}'] = (self.steps)
        new_hist[f'loss_{split}'] = (loss_metrics['loss'] / (step+1))

        for k in new_hist:
            self.hist[k].append(new_hist[k])

        if wandb:
            wandb.log(new_hist)


        self.hist[f'loss_{split}'].append(loss_metrics[f'loss'] / (step+1))

        return y_preds

    def save_hist(self, prefix='', postfix=''):
        HIST_PATH = 'data/hist/'
        if not os.path.exists(HIST_PATH):
            os.mkdir(HIST_PATH)
        fn_hist = HIST_PATH+prefix+postfix+'.csv'
        with open(fn_hist, 'w') as fh:
            print(dict(self.hist), file=fh)
        return fn_hist

    def save_model(self, prefix='', postfix='', name_as_conf=False):
        MODEL_PATH = 'data/model/'
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        if name_as_conf:
            confi_str = str(self.config.__dict__.values()).replace("'","").replace(': ','_').replace(', ',';')
        else:
            confi_str = ''
        model_name = prefix+confi_str+postfix+'.pt'
        torch.save(self.state_dict(), MODEL_PATH+model_name)
        return MODEL_PATH+model_name

    def plot_loss(self):
        plot_loss(self.hist)

    def plot_topk(self, sets=['train', 'valid', 'test'], with_last = 2):
        plot_topk(self.hist, sets=sets, with_last = with_last)

    def plot_nte(self, last_cpt=1, dataset='Sm', include_bar=True):
        plot_nte(self.hist, dataset=dataset, last_cpt=last_cpt, include_bar=include_bar)