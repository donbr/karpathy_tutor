# Karpathy Tutor application
- leverages the wisdom of Andrej Karpathy
- uses crewAI agent framework

# a word of caution
- crewAI uses OpenAI GPT4 (the more expensive non-turbo version) by default in ways I didn't expect based on my experience with LangChain
    - crewAI continued to use ChatOpenAI even after I removed it from import statements
- if you're testing this locally I recommend disabling your OpenAI keys to avoid high bills during the initial learrning curve

# credit where credit is due

- [Andrej Karpathy](https://karpathy.ai/):  an amazing inventor and teacher
    - The [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) video was great, but paired with the [nanoGPT code](https://github.com/karpathy/nanoGPT) it's even better.
- [crewAI](https://www.crewai.io/) - a very promising multi-agent framework that is breaking new ground for those who will follow