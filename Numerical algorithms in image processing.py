import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to filter the image to get a smoother version of u0
def gaussian_filter(sigma, size):
    # Gaussian function Gsigma(x,y)
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2)), (size, size))
    # Normalization of kernel
    kernel /= np.sum(kernel)
    return kernel

def convolution(image, kernel):
    # Convolve image with the kernel
    image = np.array(image, dtype=np.float32)
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    convolved_image = np.zeros_like(image)
    padded_image = np.pad(image, pad_width=padding, mode='constant')

    for i in range(convolved_image.shape[0]):
        for j in range(convolved_image.shape[1]):
            convolved_pixel = 0.0
            for m in range(kernel_size):
                for n in range(kernel_size):
                    convolved_pixel += padded_image[i+m, j+n] * kernel[m, n]
            convolved_image[i, j] = convolved_pixel

    return convolved_image.astype(np.uint8)

# Load the image and convert it into grayscale mode
image = Image.open('E:/Chan-Vese Segmentation/1.bmp').convert('L')
image = np.array(image)

# Define the parameters for the Gaussian filter
sigma = 1.0
kernel_size = 5

# Create the Gaussian filter
gaussian_kernel = gaussian_filter(sigma, kernel_size)

# Perform the convolution
convolved_image = convolution(image, gaussian_kernel)

# Define parameters for the level set evolution
epsilon = 1.0
lambda1 = 1.0
lambda2 = 1.0
nu = 0.001 * 255 * 255
dt = 0.1
num_iterations = 200

# Define the size of the image
size = convolved_image.shape

# Define the Dirac function
def dirac(x, epsilon=1.0):
    return (1/np.pi) * (epsilon / (epsilon**2 + x**2))

# Define the Heaviside step function
def heaviside(x, epsilon=1.0):
    return 0.5 * (1 + (2/np.pi) * np.arctan(x/epsilon))

# Define the function to initialize the level set
def initialize_level_set(size):
    phi = -np.ones(size)
    phi[30:70, 30:70] = 1
    return phi

# Define the function for level set evolution
def level_set_evolution(phi, image, num_iterations, dt):
    for i in range(num_iterations):
        # Compute the gradient of phi
        grad_phi = np.gradient(phi)
        norm_grad_phi = np.sqrt(grad_phi[0]**2 + grad_phi[1]**2 + 1e-10)

        # Compute the curve (divergence of the gradient of phi divided by the norm of the gradient of phi)
        curve = np.gradient(grad_phi[0]/norm_grad_phi)[0] + np.gradient(grad_phi[1]/norm_grad_phi)[1]

        # Region for the inside of the curve C
        inside_C = np.where(phi <= 0)
        c1 = np.mean(image[inside_C])

        # Region for the outside of the curve C
        outside_C = np.where(phi > 0)
        c2 = np.mean(image[outside_C])

        # Compute the data term
        data_term = lambda1 * (image - c1)**2 - lambda2 * (image - c2)**2

        # Compute the length term
        length_term = dirac(phi) * curve

        # Compute the penalty term
        penalty_term = nu * dirac(phi)

        # Compute the cotangent
        cotan = data_term + dt*(penalty_term - length_term)

        # Update phi
        phi = phi + dt * dirac(phi) * cotan

    return phi

# Initialize the level set
phi = initialize_level_set(size)

# Level set evolution
phi_final = level_set_evolution(phi, convolved_image, num_iterations, dt)

# Plot the result
plt.figure(figsize=(8, 8))
plt.imshow(convolved_image, cmap='gray')
plt.contour(phi_final, [0], colors='r')
plt.title('Segmentation with level sets')
plt.show()