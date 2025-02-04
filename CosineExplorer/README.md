---

# CosineExplorer

CosineExplorer is a Python tool that allows users to explore and visualize the cosine similarity between two 2D vectors. The program provides an interactive interface to input two vectors, calculates their cosine similarity, and then visualizes these vectors on a 2D plane.

## Features

- **Cosine Similarity Calculation**: Computes the cosine similarity between two 2D vectors.
- **Dynamic Vector Plotting**: Visualizes two vectors in a 2D space with adjustable axis limits based on the vectors' values.
- **User Input**: Allows users to input the coordinates of two vectors interactively.
- **Similarity Insights**: Displays information about the level of similarity between the vectors based on the cosine similarity value.
- **Grid and Axis Customization**: Fine-tuned gridlines and axis settings for better visualization.

## Requirements

The following Python packages are required to run the application:

- `numpy` (version 2.2.2 or higher)
- `matplotlib` (version 3.10.0 or higher)
- Python 3.12 or higher

You can install the required dependencies using `pip`:

```bash
pip install numpy matplotlib
```

## How to Run

1. Clone or download the code.
2. Install the required dependencies.
3. Run the script:

```bash
python cosineexplorer.py
```

The program will ask you to input the x and y coordinates for two vectors. After entering the vectors, it will display the cosine similarity value and a graphical representation of the vectors in 2D space.

### Example

```bash
Would you like to input two new vectors? (Enter 'y' to proceed)  y

Please enter the coordinates for two vectors in a 2D space.

For Vector 1, please provide the coordinates.
Enter the x-coordinate of Vector 1: 2
Enter the y-coordinate of Vector 1: 3

For Vector 2, please provide the coordinates.
Enter the x-coordinate of Vector 2: 4
Enter the y-coordinate of Vector 2: 6

Cosine Similarity: 1.00
The vectors are highly similar!
```

## Cosine Similarity

Cosine similarity is a metric used to measure how similar two vectors are. It ranges from -1 (completely opposite vectors) to 1 (identical vectors). The cosine similarity is calculated as follows:

```
Cosine Similarity = (A ⋅ B) / (|A| * |B|)
```

Where:
- `A` and `B` are the two vectors.
- `A ⋅ B` is the dot product of the two vectors.
- `|A|` and `|B|` are the magnitudes (or norms) of the vectors.

## License

This project is licensed under the `MIT License with Attribution` - see the [LICENSE](../LICENSE) file for details.

## Author

- **Gurucharan MK** (2025)

---
