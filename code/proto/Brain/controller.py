# ------ Linking file between Brain -> Mujoco

controller = "simple_brain" # set the controller used
load_last_controller = True

def getController(N: int, init_config=None, morphologies=None):
    if controller == "simple_brain":
        from simple_brain import get_simplebrain_controller, init_simplebrain_controllers
        init_simplebrain_controllers(N, init_config, morphologies)
        return get_simplebrain_controller()
    else:
        print("No controller defined.")
        return None
