public class BigBang
{
	private Dictionary<string, List<string>> methodDependency;
	
	public BigBang(Dictionary<string, List<string>> methodDependency)
	{
		this.methodDependency = methodDependency;
	}
	
	public List<List<string>> ClusterFunctions()
	{
		var clusters = new List<List<string>>();

		foreach (var entry in methodDependency)
		{
			var function = entry.Key;
			var dependencies = entry.Value;

			bool added = false;

			foreach (var cluster in clusters)
			{
				if (dependencies.Exists(d => cluster.Contains(d)))
				{
					cluster.Add(function);
					added = true;
					break;
				}
			}

			if (!added)
			{
				var newCluster = new List<string> { function };
				clusters.Add(newCluster);
			}
		}

		return clusters;
	}
	
	public void printClusters(List<List<string>> clusters)
	{
		foreach (List<string> cluster in clusters)
		{
			foreach (string method in cluster)
			{
				Console.WriteLine(method);
			}
			Console.WriteLine("=============================");
		}
	}
}