import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { ImageWithFallback } from "./figma/ImageWithFallback";
import { 
  Zap, 
  DollarSign, 
  Star, 
  Play,
  TrendingUp,
  Crown,
  Flame
} from "lucide-react";

interface Game {
  id: number;
  name: string;
  provider: string;
  image: string;
  jackpot?: string;
  rtp?: string;
  popular?: boolean;
  new?: boolean;
  hot?: boolean;
}

export default function GameSections() {
  const slots: Game[] = [
    {
      id: 1,
      name: "Mega Fortune",
      provider: "NetEnt",
      image: "https://images.unsplash.com/photo-1596838132731-3301c3fd4317?w=300&h=200&fit=crop",
      jackpot: "$2.4M",
      rtp: "96.4%",
      popular: true
    },
    {
      id: 2,
      name: "Dragon's Gold",
      provider: "Pragmatic Play",
      image: "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=300&h=200&fit=crop",
      rtp: "96.7%",
      new: true,
      hot: true
    },
    {
      id: 3,
      name: "Book of Ra",
      provider: "Novomatic",
      image: "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=300&h=200&fit=crop",
      rtp: "95.1%",
      popular: true
    },
    {
      id: 4,
      name: "Starburst",
      provider: "NetEnt",
      image: "https://images.unsplash.com/photo-1596838132731-3301c3fd4317?w=300&h=200&fit=crop",
      rtp: "96.1%"
    }
  ];

  const rouletteGames: Game[] = [
    {
      id: 1,
      name: "European Roulette",
      provider: "Evolution Gaming",
      image: "https://images.unsplash.com/photo-1522069169874-c58ec4b76be5?w=300&h=200&fit=crop",
      rtp: "97.3%",
      popular: true
    },
    {
      id: 2,
      name: "Lightning Roulette",
      provider: "Evolution Gaming",
      image: "https://images.unsplash.com/photo-1522069169874-c58ec4b76be5?w=300&h=200&fit=crop",
      rtp: "97.1%",
      hot: true
    },
    {
      id: 3,
      name: "American Roulette",
      provider: "Playtech",
      image: "https://images.unsplash.com/photo-1522069169874-c58ec4b76be5?w=300&h=200&fit=crop",
      rtp: "94.7%"
    }
  ];

  const crapsGames: Game[] = [
    {
      id: 1,
      name: "Live Craps",
      provider: "Evolution Gaming",
      image: "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300&h=200&fit=crop",
      rtp: "98.6%",
      popular: true
    },
    {
      id: 2,
      name: "Craps Premium",
      provider: "Playtech",
      image: "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300&h=200&fit=crop",
      rtp: "98.3%"
    }
  ];

  const renderGameCard = (game: Game) => (
    <Card key={game.id} className="group overflow-hidden bg-card/50 border-border hover:bg-card/80 transition-all duration-300 hover:scale-105">
      <div className="relative">
        <ImageWithFallback
          src={game.image}
          alt={game.name}
          className="w-full h-40 object-cover"
        />
        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
          <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90">
            <Play className="w-4 h-4 mr-2" />
            Play Now
          </Button>
        </div>
        
        {/* Badges */}
        <div className="absolute top-2 left-2 flex flex-col gap-1">
          {game.popular && (
            <Badge className="bg-primary text-primary-foreground">
              <Crown className="w-3 h-3 mr-1" />
              Popular
            </Badge>
          )}
          {game.new && (
            <Badge variant="secondary" className="bg-blue-500 text-white">
              <Zap className="w-3 h-3 mr-1" />
              New
            </Badge>
          )}
          {game.hot && (
            <Badge variant="destructive" className="bg-red-500 text-white">
              <Flame className="w-3 h-3 mr-1" />
              Hot
            </Badge>
          )}
        </div>

        {game.jackpot && (
          <div className="absolute top-2 right-2">
            <Badge className="bg-amber-500 text-black">
              <DollarSign className="w-3 h-3 mr-1" />
              {game.jackpot}
            </Badge>
          </div>
        )}
      </div>
      
      <div className="p-4">
        <h3 className="font-semibold mb-1">{game.name}</h3>
        <p className="text-sm text-muted-foreground mb-2">{game.provider}</p>
        
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 text-sm text-muted-foreground">
            <TrendingUp className="w-3 h-3" />
            RTP {game.rtp}
          </div>
          <Button size="sm" variant="outline">
            <Star className="w-3 h-3" />
          </Button>
        </div>
      </div>
    </Card>
  );

  const renderGameSection = (title: string, games: Game[], icon: React.ReactNode) => (
    <div className="mb-8">
      <div className="flex items-center gap-3 mb-6">
        {icon}
        <h2 className="text-2xl font-bold">{title}</h2>
        <Badge variant="secondary" className="ml-auto">
          {games.length} games
        </Badge>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {games.map(renderGameCard)}
      </div>
    </div>
  );

  return (
    <div className="space-y-8">
      {renderGameSection(
        "Slots",
        slots,
        <div className="w-6 h-6 bg-primary rounded flex items-center justify-center">
          <Zap className="w-4 h-4 text-primary-foreground" />
        </div>
      )}
      
      {renderGameSection(
        "Roulette",
        rouletteGames,
        <div className="w-6 h-6 bg-red-500 rounded-full flex items-center justify-center">
          <div className="w-3 h-3 bg-white rounded-full" />
        </div>
      )}
      
      {renderGameSection(
        "Craps",
        crapsGames,
        <div className="w-6 h-6 bg-blue-500 rounded flex items-center justify-center">
          <div className="w-2 h-2 bg-white rounded-full mr-0.5" />
          <div className="w-2 h-2 bg-white rounded-full" />
        </div>
      )}
    </div>
  );
}