import os,sys

class SchedulerRegistry:
    def __init__(self):
        self._schedulers = {}
    
    def register(self, name=None):
        def decorator(cls):
            scheduler_name = name or getattr(cls, 'NAME', cls.__name__)
            self._schedulers[scheduler_name] = lambda config: cls(**config)
            return cls
        return decorator
    
    def get(self, name):
        if name in self._schedulers:
            return self._schedulers[name]
        return None
    
    def list_registered(self):
        return list(self._schedulers.keys())

# Create a global instance of the registry
scheduler_registry = SchedulerRegistry()

def get_lr_scheduler( name, scheduler_config ):
    """
    Factory function to create learning rate schedulers.
    
    Args:
        name (str): Name of the scheduler to create
        scheduler_config (dict): Configuration parameters for the scheduler
        
    Returns:
        The initialized scheduler object
        
    Raises:
        ValueError: If the requested scheduler is not found
    """
    print('-----------------------------------------------')
    print("Loading scheduler with name=",name)
    print("scheduler params:")
    print(scheduler_config)
    print('-----------------------------------------------')


    factory_func = scheduler_registry.get(name)
    if factory_func:
        return factory_func(scheduler_config)
    else:
        print(f'Scheduler with name={name} not found. Registered options:')
        for kname in scheduler_registry.list_registered():
            print("  ", kname)
        raise ValueError('Scheduler not found.')



