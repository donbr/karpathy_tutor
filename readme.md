# Karpathy Tutor application
- leverages the wisdom of Andrej Karpathy
- uses crewAI agent framework

# a word of caution
- crewAI uses OpenAI GPT4 (not turbo) by default in sometimes unexpected ways
    - it will use ChatOpenAI even if it isn't included in import statements
- if you're testing this locally I recommend disabling your OpenAI keys to avoid high bills during the initial learrning curve