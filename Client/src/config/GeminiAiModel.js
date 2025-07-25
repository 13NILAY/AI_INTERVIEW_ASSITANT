import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from "@google/generative-ai";


const apiKey = "AIzaSyCgi8SoVuX1BuZwRfZEVBl6P_DyCd1gAuk";  // Vite environment variable
const genAI = new GoogleGenerativeAI(apiKey);

const model = genAI.getGenerativeModel({
  model: "gemini-2.0-flash",
});

const generationConfig = {
  temperature: 1,
  topP: 0.95,
  topK: 64,
  maxOutputTokens: 8192,
  responseMimeType: "text/plain",
};

export const chatSession = model.startChat({
  generationConfig,
});
