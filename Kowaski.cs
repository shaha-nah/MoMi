using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace MoMi;
class Kowalski
{
	private List<Blueprint> blueprints;
	private List<string> methodList;
	private Dictionary<string, List<string>> methodDependency;
	
	public Kowalski()
	{
		blueprints = new List<Blueprint>();
		methodList = new List<string>();
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
		
		// calculate structural similarity
		StructuralSimilarity structuralSimilarity = new StructuralSimilarity(methodList, methodDependency);
		int[,] structuralSimilarityMatrix = structuralSimilarity.ComputeSimiarityMatrix();
		
		// calculate semantic similarity
	}
}