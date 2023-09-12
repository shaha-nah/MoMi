using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Newtonsoft.Json.Linq;

namespace MoMi;
class Kowalski
{
	private List<Blueprint> blueprints;
	private List<string> methodList;
	private Dictionary<string, List<string>> methodDependency;
	private List<string> data;
	
	public Kowalski()
	{
		blueprints = new List<Blueprint>();
		methodList = new List<string>();
		methodDependency = new Dictionary<string, List<string>>();
		data = new List<string>();
	}
	
	public void Analysis(string folderPath)
	{
		Console.WriteLine("Performing static code analysis");
		foreach (string filePath in Directory.EnumerateFiles(folderPath, "*.cs", SearchOption.AllDirectories))
		{
			string code = File.ReadAllText(filePath);
			
			SyntaxTree syntaxTree = CSharpSyntaxTree.ParseText(code);
			CompilationUnitSyntax root = syntaxTree.GetCompilationUnitRoot();
			
			List<ClassDeclarationSyntax> classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>().ToList();
			
			foreach (ClassDeclarationSyntax classDeclaration in classDeclarations)
			{
				string className = classDeclaration.Identifier.ToString();
				
				List<MethodDeclarationSyntax> methodDeclaration = classDeclaration.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
				foreach(MethodDeclarationSyntax method in methodDeclaration)
				{
					string methodName = method.Identifier.Text;
					
					List<string> invocationNodes = method.DescendantNodes().OfType<InvocationExpressionSyntax>().Select(invocations => invocations.Expression.ToString()).Select(expression => expression.Split('.').Last()).ToList();
					string methodCalls = "";
					foreach (var invocation in invocationNodes)
					{
						methodCalls = string.Join(";", invocation);
					}
					
					List<string> variableNodes = method.DescendantNodes().OfType<VariableDeclaratorSyntax>().Select(variable => variable.Identifier.ValueText.ToString()).ToList();
					string variables = "";
					foreach(var variable in variableNodes)
					{
						variables = string.Join(";",variable);
					}
					
					methodList.Add(methodName);
					
					methodDependency[methodName] = invocationNodes;
					
					Blueprint blueprint = new Blueprint(methodName, className, methodCalls, variables);
					blueprints.Add(blueprint);
					data.Add($"{methodName}, {className}, {methodCalls}, {variables}");
				}
			}
		}
		
		AnalysisIntensifies();
	}
	
	public void AnalysisIntensifies()
	{
		foreach (var pair in methodDependency)
		{
			pair.Value.RemoveAll(value => !methodDependency.ContainsKey(value));
		}
		
		int methodCount = methodList.Count;
		// compute structural similarity matrix
		Console.WriteLine("Computing structural similarity matrix");
		StructuralSimilarity structuralSimilarity = new StructuralSimilarity(methodList, methodDependency);
		int[,] structuralSimilarityMatrix = structuralSimilarity.ComputeSimiarityMatrix();
		
		// compute semantic similarity
		Console.WriteLine("Computing semantic similarity matrix");
		SemanticSimilarity semanticSimilarity = new SemanticSimilarity(data);
		double[,] semanticSimilarityMatrix = semanticSimilarity.ComputeSimilarityMatrix();
		
		// weighted sum to get similarity
		double[,] similarityMatrix = new double[methodCount, methodCount];
		for (int i = 0; i < methodCount; i++)
		{
			for (int j = 0; j < methodCount; j++)
			{
				similarityMatrix[i, j] = (0.1 * structuralSimilarityMatrix[i, j]) + (0.9 * semanticSimilarityMatrix[i, j]);	
			}
		}
		
		// normalize similarity matrix
		double[,] normalizedSimilarityMatrix = new double[methodCount, methodCount];
		double[] vectorLengths = new double[methodCount];
		for (int i = 0; i < methodCount; i++)
		{
			double vectorLength = 0;
			
			for (int j = 0; j < methodCount; j++)
			{
				vectorLength += Math.Pow(similarityMatrix[i, j], 2);
			}
			vectorLengths[i] = Math.Sqrt(vectorLength);
		}
		
		for (int i = 0; i < methodCount; i++)
		{
			for (int j = 0; j < methodCount; j++)
			{
				normalizedSimilarityMatrix[i, j] = Math.Round(similarityMatrix[i, j] / vectorLengths[i], 3);
			}
		}
		
		// generate clusters
		Console.WriteLine("Clustering");
		Dbscan dbscan = new Dbscan(normalizedSimilarityMatrix);
		List<List<int>> clusters = dbscan.GenerateClusters();
		
		JObject clusterJson = new JObject();
		int clusterIndex = 0;
		
		foreach (var cluster in clusters)
		{
			Console.WriteLine(clusterIndex);
			Cluster clusterItems = new Cluster();
			
			foreach (var index in cluster)
			{
				clusterItems.Items.Add(blueprints[index]);
			}
			clusterJson.Add(clusterIndex.ToString(), JToken.FromObject(clusterItems));
			clusterIndex++;
		}
		File.WriteAllText("./clusters.json", clusterJson.ToString());
	}
}