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
		double eps = 0.2;
		
		for (int i = 0; i < 10; i++)
		{
			for (double j = 0.1; j < 1; j = Math.Round(j + 0.1, 1))
			{
				List<List<int>> cluster = DBSCAN(distanceMatrix, i, j);
				if (cluster.Count > 1)
				{
					Console.WriteLine($"{i}|{j}: {cluster.Count}");
				}
			}
		}
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
			// if (CalculateJaccardSimilarity(distanceMatrix, methodIndex, i) >= eps)
			if (distanceMatrix[methodIndex, i] >= eps)
			{
				neighbors.Add(i);
			}
		}

		return neighbors;
	}
	
	private double CalculateJaccardSimilarity(double[,] distanceMatrix, int methodIndex1, int methodIndex2)
	{
		double intersection = 0;
		double union = 0;
		
		for (int i = 0; i < distanceMatrix.GetLength(1); i++)
		{
			if (distanceMatrix[methodIndex1, i] > 0 && distanceMatrix[methodIndex2, i] > 0)
			{
				intersection++;
				union++;				
			}
			else if (distanceMatrix[methodIndex1, i] > 0 || distanceMatrix[methodIndex2, i] > 0)
			{
				union++;
			}
		}
		
		return intersection / union;
	}
}
