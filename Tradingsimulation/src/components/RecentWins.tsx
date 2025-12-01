import { Trophy, Clock } from "lucide-react";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";

interface WinData {
  id: number;
  player: string;
  game: string;
  amount: string;
  time: string;
  multiplier?: string;
}

export default function RecentWins() {
  const recentWins: WinData[] = [
    {
      id: 1,
      player: "CasinoKing",
      game: "Mega Fortune",
      amount: "$15,240",
      time: "2 mins ago",
      multiplier: "x150"
    },
    {
      id: 2,
      player: "LuckyStrike",
      game: "Dragon's Gold",
      amount: "$8,750",
      time: "5 mins ago",
      multiplier: "x87"
    },
    {
      id: 3,
      player: "VegasVibes",
      game: "Blackjack Pro",
      amount: "$2,850",
      time: "8 mins ago"
    },
    {
      id: 4,
      player: "SlotMaster",
      game: "Book of Ra",
      amount: "$4,320",
      time: "12 mins ago",
      multiplier: "x43"
    },
    {
      id: 5,
      player: "RoulettePro",
      game: "European Roulette",
      amount: "$1,950",
      time: "15 mins ago"
    }
  ];

  return (
    <div className="mb-8">
      <div className="flex items-center gap-3 mb-6">
        <Trophy className="w-6 h-6 text-primary" />
        <h2 className="text-2xl font-bold">Recent Wins</h2>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {recentWins.map((win) => (
          <Card key={win.id} className="p-4 bg-card/50 border-border hover:bg-card/80 transition-colors">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-medium truncate">{win.player}</span>
                {win.multiplier && (
                  <Badge variant="secondary" className="bg-primary/20 text-primary">
                    {win.multiplier}
                  </Badge>
                )}
              </div>
              
              <div>
                <p className="text-sm text-muted-foreground truncate">{win.game}</p>
                <p className="text-lg font-bold text-primary">{win.amount}</p>
              </div>
              
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="w-3 h-3" />
                {win.time}
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}