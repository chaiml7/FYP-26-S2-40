update public.sentiment_model_versions
set metrics = metrics || jsonb_build_object('balanced_accuracy', 0.83)
where model_version = 'balibpt/finbert-stocklens'
  and metrics->>'balanced_accuracy' is null;
