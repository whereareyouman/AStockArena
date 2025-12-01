import { Send, Users, MessageCircle } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { Badge } from "./ui/badge";

interface ChatMessage {
  id: number;
  username: string;
  message: string;
  timestamp: string;
  isWin?: boolean;
  amount?: string;
}

export default function GlobalChat() {
  const messages: ChatMessage[] = [
    {
      id: 1,
      username: "CasinoKing",
      message: "Just hit the jackpot on Mega Fortune!",
      timestamp: "2:34 PM",
      isWin: true,
      amount: "$15,240"
    },
    {
      id: 2,
      username: "LuckyPlayer",
      message: "Anyone playing roulette tonight?",
      timestamp: "2:35 PM"
    },
    {
      id: 3,
      username: "SlotMaster",
      message: "The new Dragon's Gold slot is amazing! 🔥",
      timestamp: "2:36 PM"
    },
    {
      id: 4,
      username: "VegasVibes",
      message: "Big win on blackjack! Lady luck is with me tonight",
      timestamp: "2:37 PM",
      isWin: true,
      amount: "$2,850"
    },
    {
      id: 5,
      username: "RoulettePro",
      message: "Red or black? What's your prediction?",
      timestamp: "2:38 PM"
    }
  ];

  return (
    <div className="w-80 bg-card border-l border-border h-full flex flex-col">
      {/* Chat Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageCircle className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">Global Chat</h3>
          </div>
          <div className="flex items-center gap-2">
            <Users className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">1,247 online</span>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {messages.map((message) => (
            <div key={message.id} className="group">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-primary/20 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-medium text-primary">
                    {message.username.charAt(0)}
                  </span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-sm">{message.username}</span>
                    <span className="text-xs text-muted-foreground">{message.timestamp}</span>
                    {message.isWin && (
                      <Badge variant="secondary" className="bg-primary/20 text-primary text-xs">
                        Win!
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-foreground break-words">{message.message}</p>
                  {message.amount && (
                    <div className="mt-1">
                      <span className="text-sm font-semibold text-primary">{message.amount}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      {/* Chat Input */}
      <div className="p-4 border-t border-border">
        <div className="flex gap-2">
          <Input 
            placeholder="Type your message..."
            className="flex-1"
          />
          <Button size="icon" className="bg-primary hover:bg-primary/90">
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}