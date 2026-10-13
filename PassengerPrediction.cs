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
