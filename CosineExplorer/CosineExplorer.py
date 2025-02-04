import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def plot_vectors(embedding1, embedding2):
    """Plot the two vectors in 2D space with dynamic axis."""
    plt.figure(figsize=(6, 6))
    
    # Origin for vectors
    origin = np.array([0, 0])

    # Plot vectors
    plt.quiver(*origin, *embedding1, angles='xy', scale_units='xy', scale=1, color='r', label='Vector 1')
    plt.quiver(*origin, *embedding2, angles='xy', scale_units='xy', scale=1, color='b', label='Vector 2')

    # Calculate axis limits based on the vector values
    all_vectors = np.vstack([embedding1, embedding2])
    max_range = np.max(np.abs(all_vectors))
    
    # Set dynamic axis limits
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    # Add fine gridlines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Fine gridlines with dashed style
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # Add major and minor ticks for finer control over gridlines
    plt.minorticks_on()  # Enable minor ticks
    plt.tick_params(which='minor', length=3, width=0.5, colors='black', grid_color='gray', grid_alpha=0.5)  # Minor ticks

    # Label the axes
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Display cosine similarity on the plot
    similarity = cosine_similarity(embedding1, embedding2)
    plt.title(f'Cosine Similarity: {similarity:.2f}')
    
    plt.show()


def get_user_input():
    """Get user input for two vectors in 2D space."""
    print("\nPlease enter the coordinates for two vectors in a 2D space.")

    # Initialize an empty list to store the embeddings
    embeddings = []

    # Loop to get input for both vectors
    for i in range(1, 3):
        print(f"\nFor Vector {i}, please provide the coordinates.")
        x = float(input(f"Enter the x-coordinate of Vector {i}: "))
        y = float(input(f"Enter the y-coordinate of Vector {i}: "))
        embeddings.append(np.array([x, y]))

    # Return the two points as vectors
    return embeddings[0], embeddings[1]


def main():
    """Main function to drive the program."""
    while True:
        # Ask if the user wants to continue or quit
        user_input = input("\nWould you like to input two new vectors? (Enter 'y' to proceed)  ").strip().lower()

        if user_input == 'y':
            embedding1, embedding2 = get_user_input()

            # Calculate and display cosine similarity
            similarity = cosine_similarity(embedding1, embedding2)
            print(f"\nCosine Similarity: {similarity:.2f}")

            if similarity > 0.8:
                print("The vectors are highly similar!")
            elif similarity > 0.5:
                print("The vectors are somewhat similar.")
            elif similarity > 0:
                print("The vectors have a slight similarity.")
            else:
                print("The vectors are not similar.")

            # Plot the vectors
            plot_vectors(embedding1, embedding2)

        else:
            print("Exiting this round. To input new vectors, please restart the program.")
            break

if __name__ == "__main__":
    main()
