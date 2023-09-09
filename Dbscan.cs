namespace MoMi;
class Dbscan
{
	private double[,] similarityMatrix;
	private List<Blueprint> blueprints;
	
	public Dbscan(double[,] similarityMatrix, List<Blueprint> blueprints)
	{
		this.similarityMatrix = similarityMatrix;
		this.blueprints = blueprints;
	}
	
	public void GenerateClusters()
	{
		int minPts = 2;
		double eps = 0.5;

		List<List<int>> clusters = DBSCAN(similarityMatrix, minPts, eps);

		foreach (var cluster in clusters)
		{
			Console.WriteLine("Cluster:");
			foreach (var index in cluster)
			{
				Console.WriteLine(index);
				blueprints[index].PrintBlueprint();

			}
			Console.WriteLine();
		}
	}

	private List<List<int>> DBSCAN(double[,] similarityMatrix, int minPts, double eps)
	{
		int numClasses = similarityMatrix.GetLength(0);

		List<int> visited = new List<int>();
		List<List<int>> clusters = new List<List<int>>();

		for (int i = 0; i < numClasses; i++)
		{
			if (visited.Contains(i)) continue;

			List<int> neighbors = GetNeighbors(i, similarityMatrix, eps);

			if (neighbors.Count < minPts)
			{
				visited.Add(i);
				continue;
			}

			List<int> cluster = new List<int> { i };
			clusters.Add(cluster);

			visited.Add(i);

			while (neighbors.Count > 0)
			{
				int currentClass = neighbors[0];
				neighbors.RemoveAt(0);

				if (!visited.Contains(currentClass))
				{
					List<int> currentClassNeighbors = GetNeighbors(currentClass, similarityMatrix, eps);

					if (currentClassNeighbors.Count >= minPts)
					{
						neighbors.AddRange(currentClassNeighbors);
					}
				}

				if (!cluster.Contains(currentClass))
				{
					cluster.Add(currentClass);
				}

				visited.Add(currentClass);
			}
		}

		return clusters;
	}

	private List<int> GetNeighbors(int classIndex, double[,] similarityMatrix, double eps)
	{
		List<int> neighbors = new List<int>();

		for (int i = 0; i < similarityMatrix.GetLength(0); i++)
		{
			if (similarityMatrix[classIndex, i] >= eps)
			{
				neighbors.Add(i);
			}
		}

		return neighbors;
	}
}
