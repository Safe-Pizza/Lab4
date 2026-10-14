/**
 * Filnamn: PassengerData.cs
 * Författare: Hanna Lindkvist
 * Datum: 2026-10-12
 * Kurs: DT071G
 * 
 * Beskrivning: Klass som representerar en passagerares data utifrån en CSV-fil. 
 * Innehåller egenskaper för lagring av överlevnad, passagerarklass, kön, ålder och antal föräldrar/barn ombord.
 * 
 * **/

using Microsoft.ML.Data;

namespace Lab4
{
    public class PassengerData
    {
        [LoadColumn(1)] public bool Survived { get; set; }
        [LoadColumn(2)] public float Pclass { get; set; }
        [LoadColumn(4)] public string Sex { get; set; } = "";
        [LoadColumn(5)] public float Age { get; set; }
        [LoadColumn(7)] public float Parch { get; set; }
    }
}
