from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import random
from sklearn.datasets import load_svmlight_file


def my_softmax(x):
    n = np.shape(x)[0]
    max_x, _ = torch.max(x, dim=1)
    max_x = torch.reshape(max_x, (n, 1))
    exp_x = torch.exp(x - max_x)
    p = exp_x / torch.reshape(torch.sum(exp_x, dim=1), (n, 1))
    p[p<10e-8] = 0
    return p

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class DATA_small(data.Dataset):


    def __init__(self, folder, transform=None):
        self.data, self.labels = load_svmlight_file(folder)
        self.data = self.data.toarray()
        self.labels = np.array(self.labels,np.int64)
    

        self.transform = transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.tensor(self.data[index]), torch.tensor(self.labels[index]).long()

        return data, target

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels 


class DATA(data.Dataset):


    def __init__(self, folder, transform=None):
   
    
        self.read_file(folder)  
        self.transform = transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.tensor(self.data[index]), torch.tensor(self.labels[index]).long()

        return data, target

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels 


    def read_file(self,folder):

        data = np.genfromtxt(folder, delimiter=',')
        self.data = data[:, 0:-1]
        self.labels = data[:, -1] - 1
        print(np.unique(self.labels))


class DATA_action(data.Dataset):
    ''' create action data from full data'''
    def __init__(self, dataset, ithclass, transform=None):
   
        #generate action dataset from full data
        
        rewards = np.zeros(len(dataset))
        for i in range(len(dataset)):
            data, label = dataset[i]

            if ithclass == label:
                rewards[i] = 0
            else:
                rewards[i] = 1
        
        self.rewards = rewards
        self.transform = transform
        self.dataset = dataset


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, _ =  self.dataset[index]
        rewards = torch.tensor(self.rewards[index])

        return data, rewards

    def __len__(self):
        return len(self.dataset)

class DATA_fullaction(data.Dataset):
    ''' create action data from full data'''
    def __init__(self, dataset, n_class, transform=None):
   
        #generate action dataset from full data
        data_new = []
        action_new = []
        reward_new = []
        action_true = []
        for i in range(len(dataset)):
            data, label = dataset[i]

            for j in range(n_class):
                data_new.append(data)
                action_new.append(j)
                action_true.append(label)
                if j != label:
                    reward_new.append(0.0)
                else:
                    reward_new.append(1.0)
        
        self.rewards = reward_new
        self.action = action_new
        self.data = data_new
        self.action_true = action_true


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data  =  torch.tensor(self.data[index])
        action = torch.tensor(self.action[index])

        rewards = torch.tensor(self.rewards[index])

        action_true = torch.tensor(self.action_true[index])

        return data, action, rewards, action_true

    def __len__(self):
        return len(self.data)

    def get_features(self):
        return self.data

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.rewards

    def get_action_true(self):
        return self.action_true


class DATA_policy_gradient(data.Dataset):
    ''' create action data from full data'''
    def __init__(self, features, rewards, actions, action_true, transform=None):
        
        self.rewards = rewards
        self.actions = actions
        self.features = features
        self.action_true = action_true


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data  =  torch.tensor(self.features[index])
        action = torch.tensor(self.actions[index])

        rewards = torch.tensor(self.rewards[index])
        action_true = torch.tensor(self.action_true[index])


        return data, action, rewards, action_true

    def __len__(self):
        return len(self.features)

class DATA_partial_random(data.Dataset):
    '''create partial data from full data'''
    def __init__(self, dataset, n_class, transform=None):
   
        #generate partial labeled dataset
        data1, _ = dataset[0] 
        actions = np.zeros(len(dataset)*n_class)
        rewards = np.zeros(len(dataset)*n_class)
        true_label = np.zeros(len(dataset)*n_class)
        features = np.zeros((len(dataset)*n_class, np.shape(data1)[0]))

        sampled_matrix = np.zeros((len(dataset)*n_class,n_class))
        reward__matrix = np.zeros((len(dataset)*n_class,n_class))

        for i in range(len(dataset)):
            for j in range(n_class):

                data, label = dataset[i]
                true_label[i*n_class+j] = label
                features[i*n_class+j] = data
                action = np.random.choice(n_class, 1)
                actions[i*n_class+j] = action[0]
                if action[0] == label:
                    rewards[i*n_class+j] = 1
                    reward__matrix[i * n_class:i * n_class + n_class, action] += 1
                sampled_matrix[i*n_class:i*n_class + n_class,action] += 1


        self.features = features
        self.action_true = true_label
        self.actions = actions
        self.rewards = rewards
        self.transform = transform
        self.dataset = dataset
        self.policy = (1.0/n_class)* np.ones(len(dataset)*n_class)
        self.reward__matrix = reward__matrix
        self.sampled_matrix = sampled_matrix

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data  =  torch.tensor(self.features[index])
        actions, rewards = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index])
        policy = self.policy[index]
        return data, actions, rewards, policy

    def __len__(self):
        return len(self.actions)

    def get_features(self):
        return self.features

    def get_action(self):
        return self.actions.astype(int)

    def get_reward(self):
        return self.rewards

    def get_policy(self):
        return self.policy

    def get_action_true(self):
        return self.action_true




class DATA_partial_action(data.Dataset):
    ''' create action data from partial data'''
    def __init__(self, dataset, ithclass, transform=None):
        m = len(dataset)
        # data_sample, _, _, _= dataset[0]
        # random policy cases

        features = dataset.get_features()
        actions =  dataset.get_action()
        rewards = dataset.get_reward()
        actions_true =  dataset.get_action_true()
        policy = dataset.get_policy()

        index = actions == ithclass
        data_new = features[index]
        reward_new = rewards[index]
        policy_new = policy[index]

        self.data = torch.tensor(data_new)
       
        self.rewards =torch.tensor(reward_new)
        self.transform = transform
        self.policy = torch.tensor(policy_new)
        self.dataset = dataset



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        data =  self.data[index]

        rewards = self.rewards[index]

        return data, rewards

    def __len__(self):
        return len(self.data)

    def get_policy(self):
        return self.policy

class DATA_partial_logistic(data.Dataset):
    '''create partial data from full data

       when you have a csv of data and data is small enough
    '''
    def __init__(self, data, n_class, estimated_policy = False, transform=None):

        '''
        data has the last column as labels
        '''
   
        #generate partial labeled dataset
     
        actions = np.zeros(len(data))
        rewards = np.zeros(len(data))
        policy = np.zeros(len(data))
        
        # generate covariate shifted data
        pca = PCA(n_components=1)
        projected = pca.fit_transform(data[:, 0:-1])
        mean_proj = np.mean(projected)
        min_proj = np.min(projected)
        mu = (mean_proj - min_proj)/1.5+ min_proj
        var = (mean_proj - min_proj)
        ## sampling
        data_shift = []
        label_shift = []
        for i in range(len(data)):
            select_prob = gaussian(projected[i], mu, var)
            if select_prob > 1:
                select_prob = 1
            rand = np.random.uniform(0, 1, 1)
            
            if rand < select_prob:
                data_shift.append(data[i, 0:-1])
                label_shift.append(data[i, -1])
        # learn logistic model for sampling
        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter = 1, multi_class='multinomial').fit(data_shift, label_shift)
        # sample y

        for i in range(len(data)):
            label = data[i, -1]
            # sample action
            rand = np.random.uniform(0, 1, 1)
        
            threshold = 0
            prob = clf.predict_proba(data[i, 0:-1].reshape(1, -1))
            # print(prob)
            for j in range(n_class):
            
                threshold = threshold + prob[0][j]
                
                if rand[0] < threshold:
                    action = j
                    actions[i] = action
                    if action != label:
                        rewards[i] = 1
                    policy[i] = prob[0][j]
                    break

        policy_estimate = np.zeros(len(data))
        if estimated_policy == True:
            # learn logistic model for estimating logging policy
            model = LogisticRegression(random_state=0, solver='lbfgs', max_iter =1, multi_class='multinomial').fit(data[:, 0:-1], actions)       
            logistic_policy = model.predict_proba(data[:, 0:-1])
            for i in range(len(data)):
                policy_estimate[i] = logistic_policy[0][int(actions[i])]
            # using robust classification?

            
        self.actions = actions
        self.rewards = rewards
        if estimated_policy == False:

            self.policy = policy
            print(policy)
            self.model = clf
        else:
            self.policy = policy_estimate
            self.model = model


        self.transform = transform
        self.data = data
        self.estimated_policy = estimated_policy
        self.unique_action =  np.unique(actions)

     



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data =  torch.tensor(self.data[index, 0:-1])
        if self.estimated_policy == False:

            actions, rewards, policy = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index]), self.policy[index]
        else:
          
            actions, rewards, policy = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index]), self.policy[index]
            
        return data, actions, rewards, policy

    def __len__(self):
        return len(self.actions)

    def get_model(self):
        return self.model



class DATA_partial_logistic_deep(data.Dataset):
    '''create partial data from full data

       when training using deep learning
    '''
    def __init__(self, data, n_class, model, estimated_policy = False, transform=None):

        '''
        data is another dataset class
        '''
   
        #generate partial labeled dataset
        data1, _= data[0]
        features = np.zeros((len(data)*n_class, np.shape(data1)[0]))

        actions = np.zeros(len(data)*n_class)
        rewards = np.zeros(len(data)*n_class)
        policy = np.zeros(len(data)*n_class)
        true_label = np.zeros(len(data)*n_class)
        # sample y

        for i in range(len(data)):
            # label = data[i, -1]
            # sample action
            for k in range(n_class):
                data_sample, target = data[i]

                rand = np.random.uniform(0, 1, 1)
            
                threshold = 0

                prob = my_softmax(model(data_sample))
                # print(prob)
                features[i*n_class+k] = data_sample
                true_label[i*n_class+k] = target
                for j in range(n_class):
                
                    threshold = threshold + prob[0][j]
                    
                    if rand[0] < threshold:
                        action = j
                        actions[i*n_class+k] = action
                        if action == target:
                            rewards[i*n_class+k] = 1
                        policy[i*n_class+k] = prob[0][j]
                        break
                     
            if estimated_policy == True:
                # learn logistic model for estimating logging policy
                model = LogisticRegression(random_state=0, solver='lbfgs', max_iter = 10, multi_class='multinomial').fit(data[:, 0:-1], actions)       
                logistic_policy = model.predict_proba(data[:, 0:-1])
                # using robust classification?

        self.features = features 
        self.true_label = true_label 
        self.actions = actions
        self.rewards = rewards
        if estimated_policy == False:

            self.policy = policy
        else:
            
            self.policy = logistic_policy

        self.transform = transform
        self.data = data
        # self.estimated_policy = estimated_policy
        self.unique_action =  np.unique(actions)
     



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data =  torch.tensor(self.features[index])
        
          
        actions, rewards, policy = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index]), self.policy[index]
            
        return data, actions, rewards, policy

    def __len__(self):
        return len(self.actions)
    def get_features(self):
        return self.features

    def get_policy(self):
        return self.policy
    def get_reward(self):
        return self.rewards

    def get_data(self):
        return self.data
    def get_action(self):
        return self.actions

    def get_action_true(self):
        return self.true_label


class DATA_partial_logistic_deep_gpu_soften_x_shift(data.Dataset):
    '''create partial data from full data
       when training using deep learning
    '''

    def __init__(self, data, n_class, model,alpha,beta, cum_prob,probality,size,sample_number = 1,estimated_policy=False, transform=None):

        '''
        data is another dataset class
        '''
        # generate partial labeled dataset
        data1, _ = data[0]
        features = np.zeros((size, np.shape(data1)[0]))
        actions = np.zeros(size )
        rewards = np.zeros(size )
        policy = np.zeros(size )
        true_label = np.zeros(size )
        # sample y
        x_shift_prob = np.zeros(size)
        index_list = [i for i in range(len(data))]

        for i in range(size):

            #get a x,y
            # rand = np.random.uniform(0, 1, 1)
            # p_g_r = cum_prob>rand
            # cum_list=cum_prob[p_g_r]
            # if len(cum_list) == 0:
            #     index = 0
            # else:
            #     min_pgr = min(cum_prob[p_g_r])
            #     index = np.where(cum_prob == min_pgr)[0][0]
            index = random.choices(index_list,weights=probality)
            data_sample, target = data[index]
            rand = np.random.uniform(0, 1, 1)
            
            # x_shift_prob[i] = probality[index]

            threshold = 0
            prob = my_softmax(model(data_sample.to('cuda')).detach().cpu())
            u = np.random.uniform(-0.5,0.5,1)
            a_det = np.argmax(prob,axis = 1)
            for j in range(prob.shape[1]):
                prob[0,j] = torch.tensor((1-alpha - beta*u)/(n_class-1))
            prob[0,a_det] = torch.tensor(alpha + beta*u)

                # print(prob)
            features[i] = data_sample
            true_label[i] = target
            for j in range(n_class):
                threshold = threshold + prob[0][j]
                if rand[0] < threshold:
                    action = j
                    actions[i] = action
                    if action == target:
                        rewards[i] = 1
                    policy[i] = prob[0][j]
                    break
            if estimated_policy == True:
                # learn logistic model for estimating logging policy
                model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10, multi_class='multinomial').fit(
                    data[:, 0:-1], actions)
                logistic_policy = model.predict_proba(data[:, 0:-1])
                # using robust classification?

        self.features = features
        self.true_label = true_label
        self.actions = actions
        self.rewards = rewards
        self.action_true = true_label
        self.x_shift_prob = x_shift_prob
        if estimated_policy == False:

            self.policy = policy
        else:

            self.policy = logistic_policy

        self.transform = transform
        self.data = data
        # self.estimated_policy = estimated_policy
        self.unique_action = np.unique(actions)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data = torch.tensor(self.features[index])

        actions, rewards, policy = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index]), \
                                   self.policy[index]

        return data, actions, rewards, policy

    def __len__(self):
        return len(self.actions)

    def get_features(self):
        return self.features

    def get_policy(self):
        return self.policy

    def get_reward(self):
        return self.rewards

    def get_data(self):
        return self.data

    def get_action(self):
        return self.actions

    def get_action_true(self):
        return self.true_label

class DATA_partial_logistic_deep_gpu_soften(data.Dataset):
    '''create partial data from full data

       when training using deep learning
    '''

    def __init__(self, data, n_class, model,alpha,beta, sample_number = 1,estimated_policy=False, transform=None):

        '''
        data is another dataset class
        '''

        # generate partial labeled dataset
        data1, _ = data[0]
        features = np.zeros((len(data) * sample_number, np.shape(data1)[0]))

        actions = np.zeros(len(data) * sample_number)
        rewards = np.zeros(len(data) * sample_number)
        policy = np.zeros(len(data) * sample_number)
        true_label = np.zeros(len(data) * sample_number)
        self.logging_policy_all = torch.ones(len(data),n_class)
        # sample y

        for i in range(len(data)):
            # label = data[i, -1]
            # sample action
            for k in range(sample_number):
                data_sample, target = data[i]

                rand = np.random.uniform(0, 1, 1)

                threshold = 0

                prob = my_softmax(model(data_sample.to('cuda')).detach().cpu())
        
                u = np.random.uniform(-0.5,0.5,1)
                a_det = np.argmax(prob,axis = 1)
                for j in range(prob.shape[1]):
                    prob[0,j] = torch.tensor((1-alpha - beta*u)/(n_class-1))
                prob[0,a_det] = torch.tensor(alpha + beta*u)

                self.logging_policy_all[i] = prob

                # print(prob)
                features[i * sample_number + k] = data_sample
                true_label[i * sample_number + k] = target
                for j in range(n_class):
                    threshold = threshold + prob[0][j]
                    if rand[0] < threshold:
                        action = j
                        actions[i * sample_number + k] = action
                        if action == target:
                            rewards[i * sample_number + k] = 1
                        policy[i * sample_number + k] = prob[0][j]
                        break
            if estimated_policy == True:
                # learn logistic model for estimating logging policy
                model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10, multi_class='multinomial').fit(
                    data[:, 0:-1], actions)
                logistic_policy = model.predict_proba(data[:, 0:-1])
                # using robust classification?

        self.features = features
        self.true_label = true_label
        self.actions = actions
        self.rewards = rewards
        self.action_true = true_label
        if estimated_policy == False:
            self.policy = policy
        else:

            self.policy = logistic_policy

        self.transform = transform
        self.data = data
        # self.estimated_policy = estimated_policy
        self.unique_action = np.unique(actions)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data = torch.tensor(self.features[index])

        actions, rewards, policy = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index]), \
                                   self.policy[index]

        return data, actions, rewards, policy

    def __len__(self):
        return len(self.actions)

    def get_features(self):
        return self.features

    def get_policy(self):
        return self.policy

    def get_reward(self):
        return self.rewards

    def get_data(self):
        return self.data

    def get_action(self):
        return self.actions

    def get_action_true(self):
        return self.true_label

class DATA_partial_logistic_deep_gpu(data.Dataset):
    '''create partial data from full data

       when training using deep learning
    '''

    def __init__(self, data, n_class, model, sample_number = 1,estimated_policy=False, transform=None):

        '''
        data is another dataset class
        '''

        # generate partial labeled dataset
        data1, _ = data[0]
        features = np.zeros((len(data) * sample_number, np.shape(data1)[0]))

        actions = np.zeros(len(data) * sample_number)
        rewards = np.zeros(len(data) * sample_number)
        policy = np.zeros(len(data) * sample_number)
        true_label = np.zeros(len(data) * sample_number)
        # sample y

        for i in range(len(data)):
            # label = data[i, -1]
            # sample action
            for k in range(sample_number):
                data_sample, target = data[i]

                rand = np.random.uniform(0, 1, 1)

                threshold = 0

                prob = my_softmax(model(data_sample.to('cuda')).detach().cpu())
                # print(prob)
                features[i * sample_number + k] = data_sample
                true_label[i * sample_number + k] = target
                for j in range(n_class):
                    threshold = threshold + prob[0][j]
                    if rand[0] < threshold:
                        action = j
                        actions[i * sample_number + k] = action
                        if action == target:
                            rewards[i * sample_number + k] = 1
                        policy[i * sample_number + k] = prob[0][j]
                        break
            if estimated_policy == True:
                # learn logistic model for estimating logging policy
                model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10, multi_class='multinomial').fit(
                    data[:, 0:-1], actions)
                logistic_policy = model.predict_proba(data[:, 0:-1])
                # using robust classification?

        self.features = features
        self.true_label = true_label
        self.actions = actions
        self.rewards = rewards
        self.action_true = true_label
        if estimated_policy == False:

            self.policy = policy
        else:

            self.policy = logistic_policy

        self.transform = transform
        self.data = data
        # self.estimated_policy = estimated_policy
        self.unique_action = np.unique(actions)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data = torch.tensor(self.features[index])

        actions, rewards, policy = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index]), \
                                   self.policy[index]

        return data, actions, rewards, policy

    def __len__(self):
        return len(self.actions)

    def get_features(self):
        return self.features

    def get_policy(self):
        return self.policy

    def get_reward(self):
        return self.rewards

    def get_data(self):
        return self.data

    def get_action(self):
        return self.actions

    def get_action_true(self):
        return self.true_label



class DATA_defined_prob(data.Dataset):
    '''create partial data from full data

       when training using deep learning
    '''
    def __init__(self, data, n_class, n_dim, prob, transform=None):

        '''
        data is another dataset class
        '''
   
        #generate partial labeled dataset

        features = np.zeros((n_class*len(data), n_dim))
        actions = np.zeros(n_class*len(data))
        rewards = np.zeros(n_class *len(data))
        policy = np.zeros(n_class *len(data))
        action_true = np.zeros(n_class *len(data))
        
        # sample y

        for i in range(len(data)):
            # label = data[i, -1]
            # sample action
            data_sample, target = data[i]
            for j in range(n_class):
                rand = np.random.uniform(0, 1, 1)
            
                threshold = 0
                # print(prob)
               
                features[n_class *i+j] = data_sample

                action_true[n_class *i+j] = target
                for k in range(n_class):
                
                    threshold = threshold + prob[k]
                    
                    if rand[0] < threshold:
                     
                        actions[n_class *i+j] = k
                        policy[n_class *i+j] = prob[k]
                        if k == target:
                            rewards[n_class *i+j] = 1
                        # policy[i] = prob[0][j]
                        break
            
        self.actions = actions
        self.rewards = rewards
        self.policy = policy
        self.action_true = action_true
        self.transform = transform
        self.data = features
        # self.estimated_policy = estimated_policy
        self.unique_action =  np.unique(actions)
        # print(self.unique_action)
     

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data =  torch.tensor(self.data[index])
    
        actions, rewards = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index])
           
        policy, action_true =  torch.tensor(self.policy[index]), torch.tensor(self.action_true[index]).long()
           
        return data, actions, rewards, action_true

    def __len__(self):
        return len(self.data)


    def get_features(self):
        return self.data

    def get_action(self):
        return self.actions.astype(int)

    def get_reward(self):
        return self.rewards

    def get_action_true(self):
        return self.action_true.astype(int)
def sample_action_batch(p, n=1, items=None):
    s = p.cumsum(axis=1)
    r = np.random.rand(p.shape[0], n, 1)
    q = np.expand_dims(s, 1) >= r
    k = q.argmax(axis=-1)
    if items is not None:
        k = np.asarray(items)[k]
    k = k.reshape(len(k))
    return k
class DATA_defined_prob_eval(data.Dataset):
    '''create partial data from full data

       when training using deep learning
    '''
    def __init__(self, data, n_class, n_dim, prob,sample_number = 1, transform=None):

        '''
        data is another dataset class
        '''
   
        #generate partial labeled dataset



        x = np.ones((len(data)*sample_number,len(data[0][0])))
        y = np.ones((len(data)*sample_number))
        for i in range(len(data)):
            data_sample, target = data[i]
            for j in range(sample_number):
                x[i*sample_number + j] = data_sample
                y[i*sample_number + j] = target
        

        features = np.zeros((sample_number*len(data), n_dim))
        actions = np.zeros(sample_number*len(data))
        rewards = np.zeros(sample_number *len(data))
        policy = np.zeros(sample_number *len(data))
        action_true = np.zeros(sample_number *len(data))
        sampled_matrix = np.zeros((len(data)*sample_number,n_class))
        reward__matrix = np.zeros((len(data)*sample_number,n_class))

        probability = np.array(prob).reshape(1,n_class).repeat(len(data)*sample_number,axis = 0)
        
        action = sample_action_batch(probability)
        features = x
        rewards = np.array((action==y).astype(int), dtype=np.double)
        policy = probability[np.arange(len(data)*sample_number),action]
        action_true = y
        actions = action
        
        self.actions = actions
        self.rewards = rewards
        self.policy = policy
        self.action_true = action_true
        self.transform = transform
        self.data = features
        self.unique_action =  np.unique(actions)
        self.reward__matrix = reward__matrix
        self.sampled_matrix = sampled_matrix
     

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data =  torch.tensor(self.data[index])
    
        actions, rewards = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index])
           
        policy =  self.policy[index]
        return data, actions, rewards, policy

    def __len__(self):
        return len(self.data)


    def get_features(self):
        return self.data

    def get_action(self):
        return self.actions.astype(int)

    def get_reward(self):
        return self.rewards

    def get_action_true(self):
        return self.action_true.astype(int)

    def get_policy(self):
        return self.policy

class DATA_defined_prob_norep(data.Dataset):
    '''
    '''
    def __init__(self, data, n_class, n_dim, prob, transform=None):

        '''
        data is another dataset class
        '''
   
        #generate partial labeled dataset

        features = np.zeros((len(data), n_dim))
        actions = np.zeros(len(data))
        rewards = np.zeros(len(data))
        policy = np.zeros(len(data))
        action_true = np.zeros(len(data))
        
        # sample y

        for i in range(len(data)):
            # label = data[i, -1]
            # sample action
            data_sample, target = data[i]
            

         
            rand = np.random.uniform(0, 1, 1)
        
            threshold = 0
            # print(prob)
           
            features[i] = data_sample
          
            action_true[i] = target
            for k in range(n_class):
            
                threshold = threshold + prob[k]
                
                if rand[0] < threshold:
                 
                    actions[i] = k
                    policy[i] = prob[k]
                    if k == target:
                        rewards[i] = 1
                    # policy[i] = prob[0][j]
                    break
            
        self.actions = actions
        self.rewards = rewards
        self.policy = policy
        self.action_true = action_true
        self.transform = transform
        self.data = features
        # self.estimated_policy = estimated_policy
        self.unique_action =  np.unique(actions)
        print(self.unique_action)
     

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data =  torch.tensor(self.data[index])
    
        actions, rewards = torch.tensor(self.actions[index]).long(), torch.tensor(self.rewards[index])
           
        policy, action_true =  torch.tensor(self.policy[index]), torch.tensor(self.action_true[index]).long()
           
        return data, actions, rewards, action_true

    def __len__(self):
        return len(self.data)


    def get_features(self):
        return self.data

    def get_action(self):
        return self.actions.astype(int)

    def get_reward(self):
        return self.rewards

    def get_action_true(self):
        return self.action_true.astype(int)

class DATA_learn_policy(data.Dataset):


    def __init__(self, data):
   
    
        self.data = data 


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target, _, _, = self.data[index]

        return data, target

    def __len__(self):
        return len(self.data)


class DATA_learn_xshift(data.Dataset):


    def __init__(self, x_train,x_test):
        train_x = np.concatenate((x_train,x_test),axis = 0)
        train_y = np.concatenate((np.ones(len(x_train),dtype = int),np.zeros(len(x_test),dtype = int)),axis = 0)
        self.data = train_x
        self.target = train_y


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        

        return self.data[index],self.target[index]

    def __len__(self):
        return len(self.data)





        
       






