namespace MoMi;
class StructuralSimilarity
{
	private List<string> methodList;
	private Dictionary<string, List<string>> methodDependency;
	
	public StructuralSimilarity(List<string> methodList, Dictionary<string, List<string>> methodDependency)
	{
		this.methodList = methodList;
		this.methodDependency = methodDependency;
	}
	
	public int[,] ComputeSimiarityMatrix()
	{
		int[,] similarityMatrix = new int[methodList.Count, methodList.Count];
		
		foreach (var pair in methodDependency)
		{
			foreach (var value in pair.Value)
			{
				similarityMatrix[methodList.IndexOf(pair.Key), methodList.IndexOf(value)] = 1;
				similarityMatrix[methodList.IndexOf(value), methodList.IndexOf(pair.Key)] = 1;
			}
		}
		return similarityMatrix;
	}
}