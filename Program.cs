/**
 * Filnamn: Program.cs
 * Författare: Hanna Lindkvist
 * Datum: 2026-10-12
 * Kurs: DT071G
 * 
 * Beskrivning: Programmet visar exempel där ML.NET används för att träna och utvärdera en maskininlärningsmodell som ska förutsäga om en
 * passagerare på Titanic överlevde eller inte baserat på olika egenskaper.
 * 
 * **/

using Lab4;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

// Skapa MLContext objekt med startvärde
MLContext mlContext = new MLContext(seed: 1);

//Läs in data från CSV-fil
string csvPath = "Titanic-Dataset.csv";
if (!File.Exists(csvPath)) //Kontrollera om filen finns
{
    Console.WriteLine($"Couldn't find data file: {Path.GetFullPath(csvPath)}");
    return;
}

//Läs in CSV-filen och skapa en IDataView för ML.NET utifrån PassengerData-klassen
IDataView data = mlContext.Data.LoadFromTextFile<PassengerData>(
    csvPath,
    hasHeader: true,
    separatorChar: ',',
    allowQuoting: true
);

//Omvandla och skapa lista från IDataView till en lista av PassengerData-objekt
List<PassengerData> passengers = mlContext.Data.CreateEnumerable<PassengerData>(data, reuseRowObject: false).ToList();
if (passengers.Count == 0) //Kontrollera om listan innehåller några passagerare
{
    Console.WriteLine("The CSV file could not be parsed or contained no passengers.");
    return;
}

//Skriv ut antal passagerare lästa från CSV-filen
Console.WriteLine($"Read {passengers.Count} passengers from {csvPath}");
Console.WriteLine();

// Förhandsvisning av de första 5 raderna i datasetet
var preview = data.Preview(maxRows: 5);
Console.WriteLine("The first 5 rows of the dataset:");
foreach (var row in preview.RowView)
{
    foreach (var column in row.Values)
    {
        Console.Write($"{column.Key}: {column.Value}\t");
    }
    Console.WriteLine();
}
Console.WriteLine();

Console.ForegroundColor = ConsoleColor.Magenta;

//Skapa en första pipleline för att förbereda data och träna modellen
var pipeline =
    mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "SexEncoded",
        inputColumnName: nameof(PassengerData.Sex))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "AgeFilled",
        inputColumnName: nameof(PassengerData.Age),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "ParentChildFilled",
        inputColumnName: nameof(PassengerData.Parch),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "PclassFilled",
        inputColumnName: nameof(PassengerData.Pclass),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "FareFilled",
        inputColumnName: nameof(PassengerData.Fare),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
    .Append(mlContext.Transforms.Concatenate(
        "Features",
        "AgeFilled",
        "SexEncoded",
        "ParentChildFilled",
        "PclassFilled",
        "FareFilled"
    ))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(
        mlContext.BinaryClassification.Trainers
            .SdcaLogisticRegression(
                labelColumnName: nameof(PassengerData.Survived),
                featureColumnName: "Features"
            )
    );

// Validate with 5-fold cross validation.
var cvResults = mlContext.BinaryClassification.CrossValidate(
    data,
    pipeline,
    numberOfFolds: 5,
    labelColumnName: nameof(PassengerData.Survived)
);

double avgAccuracy = cvResults.Average(r => r.Metrics.Accuracy);
double avgAuc = cvResults.Average(r => r.Metrics.AreaUnderRocCurve);

Console.WriteLine($"Average Accuracy (5-fold CV): {avgAccuracy:P2}");
Console.WriteLine($"Average AUC (5-fold CV): {avgAuc:P2}");

// skapa en train/test split (80/20) för utvärdering
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
// Räkna rader och skriv ut antal för varje del
long trainCount = mlContext.Data.CreateEnumerable<PassengerData>(split.TrainSet, reuseRowObject: false).LongCount();
long testCount = mlContext.Data.CreateEnumerable<PassengerData>(split.TestSet, reuseRowObject: false).LongCount();
Console.WriteLine();
Console.WriteLine($"Train/Test split: {trainCount} train rows, {testCount} test rows (80/20)");

// Träna på träningsdelen av datan
var modelOnTrain = pipeline.Fit(split.TrainSet);
// Testa på testdelen av datan
var testPredictions = modelOnTrain.Transform(split.TestSet);
// Utvärderar modellens förutsägelse med de faktiskta resultaten från överlevnadskolumnen
var testMetrics = mlContext.BinaryClassification.Evaluate(
    testPredictions,
    labelColumnName: nameof(PassengerData.Survived)
);
Console.WriteLine();
Console.WriteLine("Evaluation on 80/20 train-test split:");
Console.WriteLine($"Accuracy: {testMetrics.Accuracy:P2}"); //Andelen korrekta förutsägelser
Console.WriteLine($"AUC: {testMetrics.AreaUnderRocCurve:P2}"); //Modellens förmåga att skilja mellan klasserna
Console.WriteLine($"F1 Score: {testMetrics.F1Score:P2}"); // Sammanvägning av precision och recall
Console.WriteLine("Confusion matrix (on test set):");
Console.WriteLine(testMetrics.ConfusionMatrix.GetFormattedConfusionTable());

//
// Train the model on all data
// The cross validation are used to evaluate how well the model performs
var model = pipeline.Fit(data);

Console.ResetColor();

Console.ForegroundColor = ConsoleColor.Cyan;

//Skapa en andra pipleline för att förbereda data och träna modellen
var pipeline2 =
    mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "SexEncoded",
        inputColumnName: nameof(PassengerData.Sex))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "AgeFilled",
        inputColumnName: nameof(PassengerData.Age),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "ParentChildFilled",
        inputColumnName: nameof(PassengerData.Parch),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "PclassFilled",
        inputColumnName: nameof(PassengerData.Pclass),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "FareFilled",
        inputColumnName: nameof(PassengerData.Fare),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "EmbarkedEncoded",
        inputColumnName: nameof(PassengerData.Embarked))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "EmbarkedFilled",
        inputColumnName: "EmbarkedEncoded",
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue))
    .Append(mlContext.Transforms.Concatenate(
        "Features",
        "SexEncoded",
        "AgeFilled",
        "PclassFilled",
        "FareFilled",
        "EmbarkedFilled"
    ))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(
        mlContext.BinaryClassification.Trainers
            .SdcaLogisticRegression(
                labelColumnName: nameof(PassengerData.Survived),
                featureColumnName: "Features"
            )
    )
    );

// Validate with 5-fold cross validation.
var cvResults2 = mlContext.BinaryClassification.CrossValidate(
    data,
    pipeline2,
    numberOfFolds: 5,
    labelColumnName: nameof(PassengerData.Survived)
);

double avgAccuracy2 = cvResults2.Average(r => r.Metrics.Accuracy);
double avgAuc2 = cvResults2.Average(r => r.Metrics.AreaUnderRocCurve);

Console.WriteLine($"Average Accuracy (5-fold CV): {avgAccuracy2:P2}");
Console.WriteLine($"Average AUC (5-fold CV): {avgAuc2:P2}");

// Create a train/test split (80/20) for a evaluation
var split2 = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
// Count rows in each partition for informational purposes
long trainCount2 = mlContext.Data.CreateEnumerable<PassengerData>(split2.TrainSet, reuseRowObject: false).LongCount();
long testCount2 = mlContext.Data.CreateEnumerable<PassengerData>(split2.TestSet, reuseRowObject: false).LongCount();
Console.WriteLine();
Console.WriteLine($"Train/Test split: {trainCount2} train rows, {testCount2} test rows (80/20)");

//
// Train on the training partition and evaluate on the test partition
var modelOnTrain2 = pipeline2.Fit(split2.TrainSet);
var testPredictions2 = modelOnTrain2.Transform(split2.TestSet);
var testMetrics2 = mlContext.BinaryClassification.Evaluate(
    testPredictions2,
    labelColumnName: nameof(PassengerData.Survived)
);
Console.WriteLine();
Console.WriteLine("Evaluation on 80/20 train-test split:");
Console.WriteLine($"Accuracy: {testMetrics2.Accuracy:P2}");
Console.WriteLine($"AUC: {testMetrics2.AreaUnderRocCurve:P2}");
Console.WriteLine($"F1 Score: {testMetrics2.F1Score:P2}");
Console.WriteLine("Confusion matrix (on test set):");
Console.WriteLine(testMetrics2.ConfusionMatrix.GetFormattedConfusionTable());

//
// Train the model on all data
// The cross validation are used to evaluate how well the model performs
var model2 = pipeline2.Fit(data);

Console.ResetColor();

Console.ForegroundColor = ConsoleColor.Green;

// Create a new test passenger for prediction
PassengerData newPassenger = new()
{
    Age = 23f,
    Sex = "female",
    Pclass = 2f,
    Parch = 0f,
    Fare = 6f,
    Embarked = "S"
};

//
// Create a prediction engine
var predictionEngine =
    mlContext.Model.CreatePredictionEngine<PassengerData, PassengerPrediction>(model);

//
// Predict survival for the created test passenger
PassengerPrediction prediction = predictionEngine.Predict(newPassenger);

// Create a prediction engine
var predictionEngine2 =
    mlContext.Model.CreatePredictionEngine<PassengerData, PassengerPrediction>(model2);

//
// Predict survival for the created test passenger
PassengerPrediction prediction2 = predictionEngine2.Predict(newPassenger);

Console.WriteLine();
Console.WriteLine("Ny passagerare:");
Console.WriteLine($"Ålder: {newPassenger.Age}");
Console.WriteLine($"Kön: {newPassenger.Sex}");
Console.WriteLine($"Klass: {newPassenger.Pclass}");
Console.WriteLine($"Antal föräldrar/barn: {newPassenger.Parch}");
Console.WriteLine($"Biljettpris: {newPassenger.Fare}");

Console.ForegroundColor = ConsoleColor.Magenta;
Console.WriteLine();
Console.WriteLine(
    prediction.Survived
        ? "Modellen tror att passageraren överlever."
        : "Modellen tror att passageraren inte överlever."
);

Console.WriteLine($"Sannolikhet: {prediction.Probability:P1} : {prediction.Score}");

Console.ResetColor();
Console.ForegroundColor = ConsoleColor.Cyan;

Console.WriteLine();
Console.WriteLine(
    prediction2.Survived
        ? "Modellen tror att passageraren överlever."
        : "Modellen tror att passageraren inte överlever."
);

Console.WriteLine($"Sannolikhet: {prediction2.Probability:P1}");

Console.ResetColor();