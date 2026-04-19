import os
try:
    import wandb
except ImportError:
    wandb = None
    print("Warning: WandB not installed. Please install it with `pip install wandb`.")

from typing import Dict, Any, Optional

class WandbLogger:
    """
    WandB Logger wrapper for experiment tracking.
    """
    def __init__(self, 
                 project_name: str, 
                 run_name: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None,
                 entity: Optional[str] = None,
                 mode: str = "online",
                 resume: str = "allow"):
        """
        Initialize WandB logger.
        
        Args:
            project_name: Name of the WandB project.
            run_name: Name of the current run.
            config: Configuration dictionary to save.
            entity: WandB entity (username or team name).
            mode: WandB mode ("online", "offline", "disabled").
            resume: Resume mode ("allow", "must", "never", "auto").
        """
        if wandb is None:
            self.run = None
            return

        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.entity = entity
        self.mode = mode
        
        # Initialize wandb
        try:
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                entity=entity,
                mode=mode,
                resume=resume
            )
            print(f"WandB initialized: {project_name}/{run_name}")
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
            self.run = None
        
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to WandB.
        
        Args:
            data: Dictionary of metrics to log.
            step: Global step number (optional).
        """
        if self.run is not None:
            wandb.log(data, step=step)
            
    def finish(self):
        """Finish the WandB run."""
        if self.run is not None:
            wandb.finish()

    def watch(self, model, log="gradients", log_freq=100):
        """Watch the model to log gradients and parameters."""
        if self.run is not None:
            wandb.watch(model, log=log, log_freq=log_freq)
