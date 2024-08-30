import json

class Config:
    def __init__(self, config_path=None, jsonData=None):
        if config_path is not None:
            with open(config_path, 'r', encoding="utf-8") as f:
                jsonData = json.load(f)
            self._jsonData = jsonData
        elif jsonData is not None:
            self._jsonData = jsonData
        else:
            raise ValueError("Either config_path or jsonData must be provided.")
        
        self.load()
        
    def load(self):
        self.name = self._jsonData["name"]
        self.model_base = self._jsonData["model_base"]
        self.checkpoint_path = self._jsonData["checkpoint_path"]
        self.dataset_path = self._jsonData["dataset_path"]
   
        self.batch_size = self._jsonData["batch_size"] if "batch_size" in self._jsonData else 8
        self.num_workers = self._jsonData["num_workers"] if "num_workers" in self._jsonData else 0
        self.num_epochs = self._jsonData["num_epochs"] if "num_epochs" in self._jsonData else None
        self.max_steps = self._jsonData["max_steps"] if "max_steps" in self._jsonData else None
        self.criterion = self._jsonData["criterion"] if "criterion" in self._jsonData else "CrossEntropyLoss"
        self.optimizer = self._jsonData["optimizer"] if "optimizer" in self._jsonData else "AdamW"
        self.scheduler = self._jsonData["scheduler"] if "scheduler" in self._jsonData else "WarmupThenCosineAnnealingLR"
        self.learning_rate = self._jsonData["learning_rate"]
        self.d_coef = self._jsonData["d_coef"] if "d_coef" in self._jsonData else 1
        self.eta_min = self._jsonData["eta_min"] if "eta_min" in self._jsonData else 0.0
        self.warmup_steps = self._jsonData["warmup_steps"] if "warmup_steps" in self._jsonData else 0
        self.logging_steps = self._jsonData["logging_steps"] if "logging_steps" in self._jsonData else 5
        if "wandb" in self._jsonData:
            self.wandb = {
                "project": self._jsonData["wandb"]["project"],
                "name": self._jsonData["wandb"]["name"],
                "tags": self._jsonData["wandb"]["tags"]
            }
        
    def save(self, path):
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(self._jsonData, f, ensure_ascii=False, indent=4)