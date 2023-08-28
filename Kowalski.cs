using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace MoMi;
class Kowalski
{
	
	private List<Blueprint> blueprints;
	private Dictionary<string, List<string>> methodDependency;
	public Kowalski()
	{
		blueprints = new List<Blueprint>();
		methodDependency = new Dictionary<string, List<string>>();
	}
	
  	public void Analysis(string folderPath)
	{
		foreach (string filePath in Directory.EnumerateFiles(folderPath, "*.cs", SearchOption.AllDirectories))
		{	
			string code = File.ReadAllText(filePath);
			
			SyntaxTree syntaxTree = CSharpSyntaxTree.ParseText(code);
			CompilationUnitSyntax root = syntaxTree.GetCompilationUnitRoot();
			
			List<ClassDeclarationSyntax> classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>().ToList();
			
			foreach (ClassDeclarationSyntax classDeclaration in classDeclarations)
			{
				string className = classDeclaration.Identifier.ToString();
				List<MethodDeclarationSyntax> methodDeclarations = classDeclaration.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
				foreach(MethodDeclarationSyntax method in methodDeclarations)
				{
					string functionName = method.Identifier.Text;
					List<string> invocations = method.DescendantNodes().OfType<InvocationExpressionSyntax>().Select(invocation => invocation.Expression.ToString()).Select(expression => expression.Split('.').Last()).ToList();
					List<string> variables = method.DescendantNodes().OfType<VariableDeclaratorSyntax>().Select(variable => variable.Identifier.ValueText.ToString()).ToList();
					
					Blueprint blueprint = new Blueprint(functionName, className, invocations, variables);
					blueprints.Add(blueprint);
					
					methodDependency[functionName] = invocations;
				}
			}
		}
		AnalysisIntensifies(methodDependency);
	}
	
	public void AnalysisIntensifies(Dictionary<string, List<string>> methodDependency)
	{
		foreach (var pair in methodDependency)
		{
			pair.Value.RemoveAll(value => !methodDependency.ContainsKey(value));
		}
		
		BigBang bigBang = new BigBang(methodDependency);
		List<List<string>> clusters = bigBang.ClusterFunctions();
		bigBang.printClusters(clusters);
	}
}