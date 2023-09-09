namespace MoMi;
class Blueprint
{
	public string methodName {get; set;} = "";
	public string className {get; set;} = "";
	public string methodCalls {get; set;} = "";
	public string variableNames {get; set;} = "";
	
	public Blueprint(string methodName, string className, string methodCalls, string variableNames)
	{
		this.methodName = methodName;
		this.className = className;
		this.methodCalls = methodCalls;
		this.variableNames = variableNames;
	}
	
	public void PrintBlueprint()
	{
		Console.WriteLine($"{this.methodName} - {this.className} - {this.methodCalls} - {this.variableNames}");
	}
	
	public void printMethodDependency()
	{
		Dictionary<string, string> graph = new Dictionary<string, string>();
		
		graph[this.methodName] = this.methodCalls;
		
		foreach (var pair in graph)
		{
			Console.WriteLine(pair.Key + ": " + pair.Value);
		}
	}
}