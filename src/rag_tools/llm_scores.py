
class LLMScores():
  """Класс содержит методы для оценки качества LLM:
  метрики на основе n-грамм (BLEU, ChrF++, METEOR, TER)
  и семантические метрики (BERTScore)
  """
    def __init__(self, first_name, last_name, birth_date):
      """ 

      Args:
          first_name (_type_): _description_
          last_name (_type_): _description_
          birth_date (_type_): _description_
      """
        self.first_name = first_name
        self.last_name = last_name
        self.birth_date = birth_date  

    def calculate_meteor_score(self, df, iter=1_000, subsample=10, conf_level=0.95):
      """_summary_

      Args:
          df (_type_): _description_
          iter (_type_, optional): _description_. Defaults to 1_000.
          subsample (int, optional): _description_. Defaults to 10.
          conf_level (float, optional): _description_. Defaults to 0.95.

      Returns:
          _type_: _description_
      """

        meteors = []
        for i, row in df.iterrows():
          reference = word_tokenize(row.answer)
          prediction = word_tokenize(row.response)
          score = meteor_score([reference], prediction)
          meteors.append(score)

        df["METEOR"] = meteors
        scores = []
        for _ in range(iter):
            scores.append(np.mean(df["METEOR"].sample(subsample)))

        lower, upper, row_mean = compute_ci(scores, conf_level=conf_level)

        return row_mean, [lower, upper], df
