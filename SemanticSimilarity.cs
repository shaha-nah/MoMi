using Microsoft.ML;
namespace MoMi;
class SemanticSimilarity
{
	private List<Blueprint> blueprints;
	
	public SemanticSimilarity(List<Blueprint> blueprints)
	{
		this.blueprints = blueprints;
	}
	
	public double[,] ComputeSimilarityMatrix()
	{
		MLContext mlContext = new MLContext();

		IDataView dataview = mlContext.Data.LoadFromEnumerable(this.blueprints);

		Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.ColumnConcatenatingTransformer> pipeline = mlContext.Transforms.Text.NormalizeText("NormalizedMethodName", "methodName")
				.Append(mlContext.Transforms.Text.TokenizeIntoWords("TokenizedMethodName", "NormalizedMethodName"))
				.Append(mlContext.Transforms.Text.RemoveDefaultStopWords("TokenizedMethodName"))
				.Append(mlContext.Transforms.Conversion.MapValueToKey("TokenizedMethodName"))
				.Append(mlContext.Transforms.Text.ProduceNgrams("TokenizedMethodName"))
				.Append(mlContext.Transforms.Text.LatentDirichletAllocation("MethodFeatures", "TokenizedMethodName", numberOfTopics: 2))
				.Append(mlContext.Transforms.Text.NormalizeText("NormalizedClassName", "className"))
				.Append(mlContext.Transforms.Text.TokenizeIntoWords("TokenizedClassName", "NormalizedClassName"))
				.Append(mlContext.Transforms.Text.RemoveDefaultStopWords("TokenizedClassName"))
				.Append(mlContext.Transforms.Conversion.MapValueToKey("TokenizedClassName"))
				.Append(mlContext.Transforms.Text.ProduceNgrams("TokenizedClassName"))
				.Append(mlContext.Transforms.Text.LatentDirichletAllocation("ClassFeatures", "TokenizedClassName", numberOfTopics: 2))
				.Append(mlContext.Transforms.Text.NormalizeText("NormalizedMethodCalls", "methodCalls"))
				.Append(mlContext.Transforms.Text.TokenizeIntoWords("TokenizedMethodCalls", "NormalizedMethodCalls"))
				.Append(mlContext.Transforms.Text.RemoveDefaultStopWords("TokenizedMethodCalls"))
				.Append(mlContext.Transforms.Conversion.MapValueToKey("TokenizedMethodCalls"))
				.Append(mlContext.Transforms.Text.ProduceNgrams("TokenizedMethodCalls"))
				.Append(mlContext.Transforms.Text.LatentDirichletAllocation("MethodCallFeatures", "TokenizedMethodCalls", numberOfTopics: 2))
				.Append(mlContext.Transforms.Text.NormalizeText("NormalizedVariableNames", "variableNames"))
				.Append(mlContext.Transforms.Text.TokenizeIntoWords("TokenizedVariableNames", "NormalizedVariableNames"))
				.Append(mlContext.Transforms.Text.RemoveDefaultStopWords("TokenizedVariableNames"))
				.Append(mlContext.Transforms.Conversion.MapValueToKey("TokenizedVariableNames"))
				.Append(mlContext.Transforms.Text.ProduceNgrams("TokenizedVariableNames"))
				.Append(mlContext.Transforms.Text.LatentDirichletAllocation("VariableNameFeatures", "TokenizedVariableNames", numberOfTopics: 2))
				.Append(mlContext.Transforms.Concatenate("Features", "MethodFeatures", "ClassFeatures", "MethodCallFeatures", "VariableNameFeatures"));

		Microsoft.ML.Data.TransformerChain<Microsoft.ML.Data.ColumnConcatenatingTransformer> transformer = pipeline.Fit(dataview);
		PredictionEngine<Blueprint, TransformedBlueprint> predictionEngine = mlContext.Model.CreatePredictionEngine<Blueprint, TransformedBlueprint>(transformer);
		
		double[,] similarityMatrix = new double[blueprints.Count, blueprints.Count];
		
		for (int i = 0; i < blueprints.Count; i++)
		{
			for (int j = 0; j < blueprints.Count; j++)
			{
				// normalise feature vector to make them unit vectors
				float bp1MethodFeature1 = predictionEngine.Predict(blueprints[i]).MethodFeatures[0];
				float bp1MethodFeature2 = predictionEngine.Predict(blueprints[i]).MethodFeatures[1];
				float bp1MethodFeature1Normalized = bp1MethodFeature1/(bp1MethodFeature1 + bp1MethodFeature2);
				float bp1MethodFeature2Normalized = bp1MethodFeature2/(bp1MethodFeature1 + bp1MethodFeature2);
				
				float bp1ClassFeature1 = predictionEngine.Predict(blueprints[i]).ClassFeatures[0];
				float bp1ClassFeature2 = predictionEngine.Predict(blueprints[i]).ClassFeatures[1];
				float bp1ClassFeature1Normalized = bp1ClassFeature1/(bp1ClassFeature1 + bp1ClassFeature2);
				float bp1ClassFeature2Normalized = bp1ClassFeature2/(bp1ClassFeature1 + bp1ClassFeature2);

				float bp1MethodCallFeature1 = predictionEngine.Predict(blueprints[i]).MethodCallFeatures[0];
				float bp1MethodCallFeature2 = predictionEngine.Predict(blueprints[i]).MethodCallFeatures[1];
				float bp1MethodCallFeature1Normalized = bp1MethodCallFeature1/(bp1MethodCallFeature1 + bp1MethodCallFeature2);
				float bp1MethodCallFeature2Normalized = bp1MethodCallFeature2/(bp1MethodCallFeature1 + bp1MethodCallFeature2);

				float bp1VariableNameFeature1 = predictionEngine.Predict(blueprints[i]).VariableNameFeatures[0];
				float bp1VariableNameFeature2 = predictionEngine.Predict(blueprints[i]).VariableNameFeatures[1];
				float bp1VariableNameFeature1Normalized = bp1VariableNameFeature1/(bp1VariableNameFeature1 + bp1VariableNameFeature2);
				float bp1VariableNameFeature2Normalized = bp1VariableNameFeature2/(bp1VariableNameFeature1 + bp1VariableNameFeature2);
				
				float bp2MethodFeature1 = predictionEngine.Predict(blueprints[j]).MethodFeatures[0];
				float bp2MethodFeature2 = predictionEngine.Predict(blueprints[j]).MethodFeatures[1];
				float bp2MethodFeature1Normalized = bp2MethodFeature1/(bp2MethodFeature1 + bp2MethodFeature2);
				float bp2MethodFeature2Normalized = bp2MethodFeature2/(bp2MethodFeature1 + bp2MethodFeature2);

				float bp2ClassFeature1 = predictionEngine.Predict(blueprints[j]).ClassFeatures[0];
				float bp2ClassFeature2 = predictionEngine.Predict(blueprints[j]).ClassFeatures[1];
				float bp2ClassFeature1Normalized = bp2ClassFeature1/(bp2ClassFeature1 + bp2ClassFeature2);
				float bp2ClassFeature2Normalized = bp2ClassFeature2/(bp2ClassFeature1 + bp2ClassFeature2);

				float bp2MethodCallFeature1 = predictionEngine.Predict(blueprints[j]).MethodCallFeatures[0];
				float bp2MethodCallFeature2 = predictionEngine.Predict(blueprints[j]).MethodCallFeatures[1];
				float bp2MethodCallFeature1Normalized = bp2MethodCallFeature1/(bp2MethodCallFeature1 + bp2MethodCallFeature2);
				float bp2MethodCallFeature2Normalized = bp2MethodCallFeature2/(bp2MethodCallFeature1 + bp2MethodCallFeature2);
				
				float bp2VariableNameFeature1 = predictionEngine.Predict(blueprints[j]).VariableNameFeatures[0];
				float bp2VariableNameFeature2 = predictionEngine.Predict(blueprints[j]).VariableNameFeatures[1];
				float bp2VariableNameFeature1Normalized = bp2VariableNameFeature1/(bp2VariableNameFeature1 + bp2VariableNameFeature2);
				float bp2VariableNameFeature2Normalized = bp2VariableNameFeature2/(bp2VariableNameFeature1 + bp2VariableNameFeature2);

				// calculate dot product
				float dotProductMethodFeature = (bp1MethodFeature1Normalized * bp2MethodFeature1Normalized) + (bp1MethodFeature2Normalized * bp2MethodFeature2Normalized);
				float dotProductClassFeature = (bp1ClassFeature1Normalized * bp2ClassFeature1Normalized) + (bp1ClassFeature2Normalized * bp2ClassFeature2Normalized);
				float dotProductMethodCallFeature = (bp1MethodCallFeature1Normalized * bp2MethodCallFeature1Normalized) + (bp1MethodCallFeature2Normalized * bp2MethodCallFeature2Normalized);
				float dotProductVariableNameFeature = (bp1VariableNameFeature1Normalized * bp2VariableNameFeature1Normalized) + (bp1VariableNameFeature2Normalized * bp2VariableNameFeature2Normalized);

				// calculate magnitude
				double bp1MagnitudeMethodFeature = Math.Sqrt((bp1MethodFeature1Normalized * bp1MethodFeature1Normalized) + (bp1MethodFeature2Normalized * bp1MethodFeature2Normalized));
				double bp2MagnitudeMethodFeature = Math.Sqrt((bp2MethodFeature1Normalized * bp2MethodFeature1Normalized) + (bp2MethodFeature2Normalized * bp2MethodFeature2Normalized));

				double bp1MagnitudeClassFeature = Math.Sqrt((bp1ClassFeature1Normalized * bp1ClassFeature1Normalized) + (bp1ClassFeature2Normalized * bp1ClassFeature2Normalized));
				double bp2MagnitudeClassFeature = Math.Sqrt((bp2ClassFeature1Normalized * bp2ClassFeature1Normalized) + (bp2ClassFeature2Normalized * bp2ClassFeature2Normalized));

				double bp1MagnitudeMethodCallFeature = Math.Sqrt((bp1MethodCallFeature1Normalized * bp1MethodCallFeature1Normalized) + (bp1MethodCallFeature2Normalized * bp1MethodCallFeature2Normalized));
				double bp2MagnitudeMethodCallFeature = Math.Sqrt((bp2MethodCallFeature1Normalized * bp2MethodCallFeature1Normalized) + (bp2MethodCallFeature2Normalized * bp2MethodCallFeature2Normalized));

				double bp1MagnitudeVariableNameFeature = Math.Sqrt((bp1VariableNameFeature1Normalized * bp1VariableNameFeature1Normalized) + (bp1VariableNameFeature2Normalized * bp1VariableNameFeature2Normalized));
				double bp2MagnitudeVariableNameFeature = Math.Sqrt((bp2VariableNameFeature1Normalized * bp2VariableNameFeature1Normalized) + (bp2VariableNameFeature2Normalized * bp2VariableNameFeature2Normalized));

				// calculate cosine similarity
				double cosineSimilarityMethodFeature = dotProductMethodFeature / (bp1MagnitudeMethodFeature * bp2MagnitudeMethodFeature);
				if (double.IsNaN(cosineSimilarityMethodFeature))
				{
					cosineSimilarityMethodFeature = 0;
				}
				double cosineSimilarityClassFeature = dotProductClassFeature / (bp1MagnitudeClassFeature * bp2MagnitudeClassFeature);
				if (double.IsNaN(cosineSimilarityClassFeature))
				{
					cosineSimilarityClassFeature = 0;
				}
				double cosineSimilarityMethodCallFeature = dotProductMethodCallFeature / (bp1MagnitudeMethodCallFeature * bp2MagnitudeMethodCallFeature);
				if (double.IsNaN(cosineSimilarityMethodCallFeature))
				{
					cosineSimilarityMethodCallFeature = 0;
				}
				double cosineSimilarityVariableNameFeature = dotProductVariableNameFeature / (bp1MagnitudeVariableNameFeature * bp2MagnitudeVariableNameFeature);
				if (double.IsNaN(cosineSimilarityVariableNameFeature))
				{
					cosineSimilarityVariableNameFeature = 0;
				}

				// average of cosine similarities
				double similarity = (cosineSimilarityMethodFeature + cosineSimilarityClassFeature + cosineSimilarityMethodCallFeature + cosineSimilarityVariableNameFeature) / 4;
				
				similarityMatrix[i, j] = similarity;
				similarityMatrix[j, i] = similarity;
			}
		}
		
		return similarityMatrix;
	}	
	
}


