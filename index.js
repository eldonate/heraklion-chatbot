import { config } from "dotenv";
config(); // Load environment variables from .env

import fs from 'fs';
import express from 'express';
import { OpenAI } from "langchain/llms/openai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Load your OpenAI API key from .env
const openAiApiKey = process.env.OPENAI_API_KEY;
if (!openAiApiKey) {
  console.error("Error: OPENAI_API_KEY not set in .env");
  process.exit(1);
}

// Read the text file
const text = fs.readFileSync('heraklion_history.txt', 'utf8');

// Split the text into smaller chunks to reduce request size
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,    // Reduced chunk size
  chunkOverlap: 50   // Reduced overlap
});
const docs = await splitter.createDocuments([text]);

// Prepare embeddings (no request is made yet)
const embeddings = new OpenAIEmbeddings({ openAIApiKey: openAiApiKey });

// Define the strict prompt template
const template = `
You are a highly accurate assistant that answers questions solely based on the provided context about Heraklion. 
Do not use any external knowledge or make assumptions beyond the context. 
If the answer to the question is not found within the context, respond with "Λυπάμαι, δεν μπορώ να σε βοηθήσω με αυτό."
All responses must be in Greek.

Here is the context:
{context}

Question: {query}
Answer:
`;

// Initialize the LLM
const llm = new OpenAI({
  openAIApiKey: openAiApiKey,
  temperature: 0,                // Set temperature to 0 for deterministic responses
  modelName: "gpt-3.5-turbo"     // Ensure you're using the desired model
});

// Initialize variables for lazy loading
let vectorStore = null;
let chain = null;

// Set up an Express server
const app = express();
app.use(express.json());

// POST /ask endpoint
app.post('/ask', async (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: "No question provided" });

  try {
    // Initialize vectorStore and chain on the first request
    if (!vectorStore) {
      vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
    }

    if (!chain) {
      chain = RetrievalQAChain.fromLLM(llm, vectorStore.asRetriever(), {
        promptTemplate: template
      });
    }

    // Retrieve and answer the question
    const response = await chain.call({ query: question });
    res.json({ answer: response.text.trim() });
  } catch (error) {
    console.error("Error processing request:", error);
    res.status(500).json({ error: "Error processing request" });
  }
});

// GET / endpoint
app.get('/', (req, res) => {
  res.send("Node.js Chatbot Server Running");
});

// Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Chatbot running on port ${port}`));
