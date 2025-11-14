export interface ToolCall {
  id: string;
  name: string;
  args: string;
  result?: string;
  status: "running" | "complete";
}

export interface Message {
  id: number;
  role: "user" | "bot" | "tool";
  content: string;
  isStreaming?: boolean;
  toolCalls?: ToolCall[];
}
