# vercel_othello_dqn_ai


```

Vercel の Lambda サイズ制限（50MB）に引っかかっているようですね。PyTorch が含まれているため、パッケージサイズが大きすぎるのが問題です。
解決のために2つのアプローチを提案します：

Pythonベースではなく、フロントエンドのみの実装に変更する（より確実）
PyTorch を排除して、シンプルなルールベースのAIに変更する（次善策）

```


```

WARN! Due to `builds` existing in your configuration file, the Build and Development Settings defined in your Project Settings will not apply. Learn More: https://vercel.link/unused-build-settings
Installing required dependencies...
Build Completed in /vercel/output [4m]
Deploying outputs...
Failed to process build result for "app.py". Data: {"type":"Lambda"}.
Error: data is too long

```


```

depoly failure 2025-02-18 14:30
Running "vercel build"
Vercel CLI 41.1.3
WARN! Due to `builds` existing in your configuration file, the Build and Development Settings defined in your Project Settings will not apply. Learn More: https://vercel.link/unused-build-settings
Installing required dependencies...
Build Completed in /vercel/output [4m]

```
