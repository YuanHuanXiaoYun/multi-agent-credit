import torch


class Parameter:
    def __init__(self,model_type='others',num_agents=10,num_tasks=40):
        self.state_size=4
        self.episode_max = 2000
        self.num_agents=num_agents
        self.num_tasks=num_tasks
        self.action_size=self.num_tasks + 1
        self.map_size=(50,50)
        self.reward_individual=False

        self.credit_variant = "ours"


        self.credit_mode = "crowd_wait_penalty_v1"
        self.credit_llm_model = ""
        self.credit_llm_ratio = 0.3          
        self.credit_llm_min_interval = 0.5   
        self.credit_gate_min_interval = 1.0 
        self.credit_llm_cache_ttl = 5.0
        self.credit_per_episode_budget = self.num_tasks

        self.credit_overlap_coeff = 0.2
        self.credit_wait_coeff = 0.001
        self.credit_overlap_peak_weight = 0.5

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.isTrain=True
        self.path_model='model'
        self.path_env='env'

        self.Q_expand=False
        self.Q_Probability=True
        self.IndeOptim=False

if __name__ == "__main__":
    mp=Parameter()
    print(mp.GNNParams.num_agents)
    

    