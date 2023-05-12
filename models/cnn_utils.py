import numpy as np

def convolution(kernel, image, s=1, p=0):

    m, n = image.shape
    k_v, k_h = kernel.shape
    m_conv = int((m - k_v + 2 * p) / s + 1)
    n_conv = int((n - k_h + 2 * p) / s + 1)
    convoluted = np.zeros((m_conv, n_conv))

    for i in range(m_conv):
        for j in range(n_conv):
            convoluted[i, j] = np.sum(kernel * image[i: i + s, j: j + s])
    
    return convoluted

def pooling(image, operation=np.max):

    result = np.empty((image.shape[0] // 2, image.shape[1] // 2))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = operation(image[i * 2 : (i * 2) + 2, j * 2 : (j * 2) + 2])

    return result


def test_me():
    
    image = np.array([[1, 1, 1, 1], 
              [0, 0, 0, 0], 
              [1, 1, 1, 1], 
              [0, 0, 0, 0]])
    
    kernel = np.array([[1, 1], [0, 0]])

    conv = convolution(kernel, image)

    print(conv)

def test_me2():
    
    image = np.array([[1, 1, 1, 1], 
              [0, 0, 0, 0], 
              [1, 1, 1, 1], 
              [0, 0, 0, 0]])
    
    kernel = np.array([[1, 1], [0, 0]])

    conv = convolution(kernel, image, p=1)

    print(conv)

def test_pooling():
    
    image = np.array([[2, 1, 1, 1], 
              [0, 0, 0, 2], 
              [3, 1, 2, 1], 
              [0, 4, 0, 3]])
    pool = pooling(image)

    print(pool)

if __name__ == "__main__":
    
    # test_me2()

    test_pooling()