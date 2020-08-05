class CounterfactualsBase:
    """
    Base class for counterfactuals ; useless for now.
    """
    def __init__(self,
                obs_to_interprete,
                prediction_fn,
                target_class=None,
                caps=None):
        """
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))
        self.target_class = target_class
        self.caps = caps
        
        def generate_layer():
            return 1
        
        def check_ennemies():
            return 1
        
        
class GrowingSpheresBase:
    #def init
    #def feature selection
    #def find counterfactual