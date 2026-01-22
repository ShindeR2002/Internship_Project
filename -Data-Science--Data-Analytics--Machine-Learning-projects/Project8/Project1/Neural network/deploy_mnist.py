import numpy as np
import os
import numpy as np
import os
from neural_network import Network 

class MNIST_Deployer:
    def __init__(self, model_folder="final_97_model"):
        # NEW: Find the absolute path to the script's directory
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.model_folder = os.path.join(base_path, model_folder)
        
        self.model = Network()
        self.model.path([784, 256, 10]) 
        self._load_weights()

    def _load_weights(self):
        fc_layer_idx = 0
        for layer in self.model.layers:
            if hasattr(layer, 'weights_'):
                # Construct the full path to the .npy files
                weight_path = os.path.join(self.model_folder, f"L{fc_layer_idx}_W.npy")
                bias_path = os.path.join(self.model_folder, f"L{fc_layer_idx}_B.npy")
                
                layer.weights_ = np.load(weight_path)
                layer.bias_ = np.load(bias_path)
                fc_layer_idx += 2 
        print(f"✅ 97.33% Accuracy weights loaded successfully from: {self.model_folder}")
    def predict_digit(self, image_data):
        # image_data should be a flattened (1, 784) normalized array
        output = self.model.predict(image_data)
        return np.argmax(output)

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    deployer = MNIST_Deployer()
    # You can now call deployer.predict_digit(new_image) anywhere!