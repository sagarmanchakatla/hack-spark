import { streamText } from "ai";
import { createOpenAI as createGroq } from "@ai-sdk/openai";
import fs from "fs"; // to read JSON files from the file system

const groq = createGroq({
  baseURL: "https://api.groq.com/openai/v1",
  apiKey: process.env.GROQ_API_KEY,
});

const formatHealthDataForAssistant = (healthData: any) => {
  return `
  Here is the user's current health data:
  - Heart Rate: ${healthData.heart_rate} bpm
  - Body Fat: ${healthData.body_fat}%
  - Weight: ${healthData.weight} kg
  - BMI: ${healthData.bmi}
  - Sleep Asleep: ${healthData.sleep_asleep} hours
  - Blood Glucose Level: ${healthData.blood_glucose_level} mg/dL
  - Hypertension: ${healthData.hypertension}
  - Heart Disease Risk: ${healthData.heart_health.risk_category} (${(
    healthData.heart_health.risk_probability * 100
  ).toFixed(2)}% risk)
  - Diabetes Status: ${healthData.diabities_info.message}
  `;
};

const loadHealthDataFromFile = () => {
  const rawData = fs.readFileSync(
    "C:/hack-spark/bot/app/data/health_data.json",
    "utf-8"
  ); // Read JSON file synchronously
  return JSON.parse(rawData); // Parse the JSON data into an object
};

const systemPrompt = `
You are a personal health assistant powered by AI. Your goal is to help users manage their health data, provide insights, and offer actionable recommendations based on their health metrics.

Key Functions:
1. Health Data Analysis: Interpret and explain health metrics like heart rate, BMI, sleep patterns, etc.
2. Personalized Recommendations: Provide tailored advice based on user's health data.
3. Risk Assessment: Highlight potential health risks and suggest preventive measures.
4. Lifestyle Guidance: Offer tips for improving diet, exercise, and sleep quality.
5. Emergency Alerts: Notify users when immediate medical attention might be needed.

Style:
- Be empathetic and supportive
- Use clear, simple language to explain health metrics
- Provide actionable, evidence-based recommendations
- Be proactive in identifying potential health risks
- Offer both immediate and long-term health strategies

Example responses:
- "Your heart rate is slightly elevated today. Let's take a deep breath and relax for a few minutes."
- "Based on your sleep data, you might benefit from going to bed 30 minutes earlier tonight."
- "Your BMI suggests you're overweight. Would you like some tips on healthy weight management?"
- "Your blood glucose levels are within the normal range. Keep up the good work!"
- "I noticed your blood pressure is high. Let's schedule a check-up with your doctor soon."
`;

export async function POST(req: Request) {
  const { messages } = await req.json();
  console.log(messages);
  // Load the health data from the file
  const healthData = loadHealthDataFromFile();

  // Format the health data
  const formattedHealthData = formatHealthDataForAssistant(healthData);

  // Add the formatted health data to the message context
  const enhancedMessages = [
    ...messages,
    {
      role: "system",
      content: `${formattedHealthData}\n\n`,
    },
  ];

  // Call the model to generate the response based on the enhanced context
  const result = await streamText({
    model: groq("llama-3.3-70b-versatile"),
    messages: enhancedMessages,
    system: systemPrompt,
  });

  // Return the result as a data stream response
  return result.toDataStreamResponse();
}
