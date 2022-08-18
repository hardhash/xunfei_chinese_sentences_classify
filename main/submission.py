import pandas as pd
import time
df = pd.read_csv('result.csv')
df1= pd.read_csv('submissionExample.csv', sep='\t')

submission = pd.DataFrame({
    'id' : df1.id
})

submission['label'] = df['label']
submission.to_csv(f'submission_{time.time()}.csv', index=False, sep='\t')