/**
 * Filnamn: PassengerPrediction.cs
 * Författare: Hanna Lindkvist
 * Datum: 2026-10-12
 * Kurs: DT071G
 *
 * Beskrivning: Klass som representerar en passagerares överlevnadsprognos. Innehåller egenskaper för lagring från ML.NET: probability, score samt en bool
 * för att indikera om passageraren förväntas överleva eller inte.
 * 
 * **/

using Microsoft.ML.Data;

namespace Lab4
{
    public class PassengerPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Survived { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
