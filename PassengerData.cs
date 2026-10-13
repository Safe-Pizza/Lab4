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
        [LoadColumn(9)] public float Fare { get; set; }
        [LoadColumn(11)] public string Embarked { get; set; }
    }
}
