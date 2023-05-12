import models.cnn_utils
import array as np

def test_me():
    
    image = np.array([[1, 1, 1, 1], 
              [0, 0, 0, 0], 
              [1, 1, 1, 1], 
              [0, 0, 0, 0]])
    
    kernel = np.array([[1, 1], [0, 0]])

    conv = models.cnn_utils.convolution(kernel, image)

    print(conv)

if __name__ == "__main__":
    
    test_me()
