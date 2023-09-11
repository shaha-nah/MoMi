namespace MoMi;
class Dbscan
{
	private double[,] distanceMatrix;
	
	public Dbscan(double[,] distanceMatrix)
	{
		this.distanceMatrix = distanceMatrix;
	}
	
	public List<List<int>> GenerateClusters()
	{
		int minPts = 3;
		double eps = 0.88;

		List<List<int>> clusters = DBSCAN(distanceMatrix, minPts, eps);

		return clusters;
	}

	private List<List<int>> DBSCAN(double[,] distanceMatrix, int minPts, double eps)
	{
		int numMethods = distanceMatrix.GetLength(0);

		List<int> visited = new List<int>();
		List<List<int>> clusters = new List<List<int>>();

		for (int i = 0; i < numMethods; i++)
		{
			if (visited.Contains(i)) continue;

			List<int> neighbors = GetNeighbors(i, distanceMatrix, eps);

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
				int currentMethod = neighbors[0];
				neighbors.RemoveAt(0);

				if (!visited.Contains(currentMethod))
				{
					List<int> currentMethodNeighbors = GetNeighbors(currentMethod, distanceMatrix, eps);

					if (currentMethodNeighbors.Count >= minPts)
					{
						neighbors.AddRange(currentMethodNeighbors);
					}
				}

				if (!cluster.Contains(currentMethod))
				{
					cluster.Add(currentMethod);
				}

				visited.Add(currentMethod);
			}
		}

		return clusters;
	}

	private List<int> GetNeighbors(int methodIndex, double[,] distanceMatrix, double eps)
	{
		List<int> neighbors = new List<int>();

		for (int i = 0; i < distanceMatrix.GetLength(0); i++)
		{
			if (distanceMatrix[methodIndex, i] >= eps)
			{
				neighbors.Add(i);
			}
		}

		return neighbors;
	}
}
