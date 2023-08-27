using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace MoMi;
class Kowalski
{
	
	private List<Blueprint> blueprints;
	public Kowalski()
	{
		blueprints = new List<Blueprint>();
	}
	
	public void Analysis(string folderPath)
	{
		foreach (string filePath in Directory.EnumerateFiles(folderPath, "*.cs", SearchOption.AllDirectories))
		{	
			string code = File.ReadAllText(filePath);
			
			SyntaxTree syntaxTree = CSharpSyntaxTree.ParseText(code);
			CompilationUnitSyntax root = syntaxTree.GetCompilationUnitRoot();
			
			List<MethodDeclarationSyntax> methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
			foreach(MethodDeclarationSyntax method in methodDeclarations)
			{
				string functionName = method.Identifier.Text;
				string className = method.Ancestors().OfType<ClassDeclarationSyntax>().FirstOrDefault()!.Identifier.Text;
				string namespaceName = method.Ancestors().OfType<NamespaceDeclarationSyntax>().FirstOrDefault()!.Name.ToString();
				List<string> invocations = method.DescendantNodes().OfType<InvocationExpressionSyntax>().Select(invocation => invocation.Expression.ToString()).ToList();
				List<string> variables = method.DescendantNodes().OfType<VariableDeclaratorSyntax>().Select(variable => variable.Identifier.ValueText.ToString()).ToList();
				
				Blueprint blueprint = new Blueprint(functionName, className, namespaceName, invocations, variables);
				blueprint.PrintBlueprint();
				blueprints.Add(blueprint);
			}
		}
	}
}