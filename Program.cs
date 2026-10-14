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
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue))
    .Append(mlContext.Transforms.Concatenate(
        "Features",
        "AgeFilled",
        "SexEncoded",
        "ParentChildFilled",
        "PclassFilled"
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



// skapa en train/test split (80/20) för utvärdering
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
// Räkna rader och skriv ut antal för varje del
long trainCount = mlContext.Data.CreateEnumerable<PassengerData>(split.TrainSet, reuseRowObject: false).LongCount();
long testCount = mlContext.Data.CreateEnumerable<PassengerData>(split.TestSet, reuseRowObject: false).LongCount();


// Träna på träningsdelen av datan
var modelOnTrain = pipeline.Fit(split.TrainSet);
// Testa på testdelen av datan
var testPredictions = modelOnTrain.Transform(split.TestSet);
// Utvärderar modellens förutsägelse med de faktiskta resultaten från överlevnadskolumnen
var testMetrics = mlContext.BinaryClassification.Evaluate(
    testPredictions,
    labelColumnName: nameof(PassengerData.Survived)
);

// Sparar den tränade modellen i variabel
var model = pipeline.Fit(data);

while (true) //Loop som körs tills användaren väljer att avsluta programmet
{
    Console.Clear();

    ShowMenu();
    int inp = (int)Console.ReadKey(true).Key; //Lagra userinput i varibel

    //switch-stats som tar emot userinput 1, 2 eller X (88)
    switch (inp)
    {
        //input 1 skapar nytt inlägg
        case (int)ConsoleKey.D1:
        case (int)ConsoleKey.NumPad1:
            Console.Clear();
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
            Console.WriteLine();
            Console.WriteLine($"Average Accuracy (5-fold CV): {avgAccuracy:P2}"); //Korsvalideringens genomsnittliga noggrannhet
            Console.WriteLine($"Average AUC (5-fold CV): {avgAuc:P2}"); //Korsvalideringens genomsnittliga AUC
            Console.WriteLine();
            Console.WriteLine($"Train/Test split: {trainCount} train rows, {testCount} test rows (80/20)"); //Antal rader i tränings- och testuppsättningen
            Console.WriteLine();
            Console.WriteLine("Evaluation on 80/20 train-test split:");
            Console.WriteLine($"Accuracy: {testMetrics.Accuracy:P2}"); //Andelen korrekta förutsägelser
            Console.WriteLine($"AUC: {testMetrics.AreaUnderRocCurve:P2}"); //Modellens förmåga att skilja mellan klasserna
            Console.WriteLine($"F1 Score: {testMetrics.F1Score:P2}"); // Sammanvägning av precision och recall
            Console.WriteLine("Confusion matrix (on test set):");
            Console.WriteLine(testMetrics.ConfusionMatrix.GetFormattedConfusionTable());

            Console.WriteLine($"\nTryck på valfri tagnet för att återgå till huvudmenyn...");
            Console.ReadKey(true);
            break;
        //input 2 ta bort inlägg med index som parameter
        case (int)ConsoleKey.D2:
        case (int)ConsoleKey.NumPad2:
            Console.CursorVisible = true;

            float ageInput = GetAgeInput();
            string sexInput = GetSexInput();
            float pclassInput = GetPclassInput();
            float parchInput = GetParchInput();

            // Skapa en ny passagerare för förutsägelse
            PassengerData newPassenger = new()
            {
                Age = ageInput,
                Sex = sexInput,
                Pclass = pclassInput,
                Parch = parchInput
            };
            // Skapa en prediktionsmotor
            var predictionEngine =
                mlContext.Model.CreatePredictionEngine<PassengerData, PassengerPrediction>(model);
            // Förutsäg överlevnad för den nya passageraren
            PassengerPrediction prediction = predictionEngine.Predict(newPassenger);
            Console.WriteLine();
            Console.WriteLine("Ny passagerare:");
            Console.WriteLine($"Ålder: {newPassenger.Age}");
            Console.WriteLine($"Kön: {newPassenger.Sex}");
            Console.WriteLine($"Klass: {newPassenger.Pclass}");
            Console.WriteLine($"Antal föräldrar/barn: {newPassenger.Parch}");
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine();
            Console.WriteLine(
                prediction.Survived
                    ? "Modellen tror att passageraren överlever."
                    : "Modellen tror att passageraren inte överlever."
            );
            Console.WriteLine($"Sannolikhet: {prediction.Probability:P1}");
            Console.ResetColor();

            Console.WriteLine($"\nTryck på valfri tangent för att återgå till huvudmenyn...");
            Console.ReadKey(true);
            break;

        //input X avslutar programmet
        case 88:
            Environment.Exit(0);
            break;
    }
}
//Metod för att visa programmets meny
static void ShowMenu()
{
    Console.CursorVisible = false;
    Console.ForegroundColor = ConsoleColor.DarkMagenta;
    //Programmets meny
    Console.WriteLine("╔════════════════════════════════════════════════╗");
    Console.WriteLine("║                 ML MED TITANIC                 ║");
    Console.WriteLine("╚════════════════════════════════════════════════╝");
    Console.ForegroundColor = ConsoleColor.White;
    Console.WriteLine("\n     [1]   Visa modellens data och testresultat");
    Console.WriteLine("     [2]   Förutsäg överlevnad");
    Console.WriteLine("     [X]   Avsluta");
    Console.WriteLine();
    Console.WriteLine();
}

//Metod för att validera ålderinput från användaren. Den accepterar endast värden mellan 0 och 100 och returnerar det som float.
static float GetAgeInput()
{
    while (true)
    {
        Console.Write("Ålder: ");
        string? ageInput = Console.ReadLine();
        if (!string.IsNullOrEmpty(ageInput) && float.TryParse(ageInput, out float age))
        {
            if (age <= 0 || age <= 100)
            {
                return age;
            }
            else
            {
                Console.WriteLine("Ålder måste vara mellan 0 och 100. Försök igen.");
            }
        }
        else
        {
            Console.WriteLine("Ogiltig ålder. Försök igen.");
        }
    }
}

//Metod för att validera könsinput från användaren. Den accepterar endast "male" eller "female" (case-insensitive) och returnerar det i lowercase.
static string GetSexInput()
{
    while (true)
    {
        Console.Write("Kön (male/female): ");
        string? sexInput = Console.ReadLine();
        if (!string.IsNullOrEmpty(sexInput))
        {
            if (string.Equals(sexInput, "male", StringComparison.OrdinalIgnoreCase) || string.Equals(sexInput, "female", StringComparison.OrdinalIgnoreCase))
            {
                return sexInput.ToLower();
            }
            else
            {
                Console.WriteLine("Ogiltigt kön. Försök igen.");
            }

        }
    }
}
//Metod för att validera biljettklassinput från användaren. Den accepterar endast "1", "2" eller "3" och returnerar det som float.
static float GetPclassInput()
{
    while (true)
    {
        Console.Write("Biljettklass (1–3): ");
        string? pclassInput = Console.ReadLine();
        if (pclassInput == "1" || pclassInput == "2" || pclassInput == "3")
        {
            return float.Parse(pclassInput);
        }
        else
        {
            Console.WriteLine("Ogiltig biljettklass. Försök igen.");
        }
    }
}
//Metod för att validera antal föräldrar/barn input från användaren. Den accepterar endast värden mellan 0 och 6 (utifrån värden i dataset) och returnerar det som float.
static float GetParchInput()
{
    while (true)
    {
        Console.Write("Antal föräldrar/barn: ");
        string? parchInput = Console.ReadLine();
        if (!string.IsNullOrEmpty(parchInput) && float.TryParse(parchInput, out float parch))
        {
            if (parch >= 0 && parch < 7)
            {
                return parch;
            }
            else
            {
                Console.WriteLine("Värde måste vara mellan 0 och 6.");
            }
        }
        else
        {
            Console.WriteLine("Ogiltigt antal föräldrar/barn. Försök igen.");
        }
    }
}