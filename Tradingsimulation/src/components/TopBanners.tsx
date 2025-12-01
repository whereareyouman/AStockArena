import { ImageWithFallback } from "./figma/ImageWithFallback";
import { Button } from "./ui/button";
import { Star, Gift } from "lucide-react";

export default function TopBanners() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      {/* Welcome Bonus Banner */}
      <div className="relative bg-gradient-to-r from-primary/20 to-primary/40 rounded-xl p-6 overflow-hidden">
        <div className="relative z-10">
          <div className="flex items-center gap-2 mb-2">
            <Gift className="w-6 h-6 text-primary" />
            <span className="text-sm font-medium text-primary">Welcome Bonus</span>
          </div>
          <h2 className="text-2xl font-bold mb-2">Get 200% Bonus</h2>
          <p className="text-muted-foreground mb-4">
            Up to $2,000 + 50 Free Spins on your first deposit
          </p>
          <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
            Claim Now
          </Button>
        </div>
        <div className="absolute -right-4 -top-4 w-32 h-32 bg-primary/20 rounded-full blur-xl" />
        <div className="absolute -right-8 -bottom-8 w-24 h-24 bg-primary/30 rounded-full blur-lg" />
      </div>

      {/* VIP Program Banner */}
      <div className="relative bg-gradient-to-r from-amber-500/20 to-amber-600/40 rounded-xl p-6 overflow-hidden">
        <div className="relative z-10">
          <div className="flex items-center gap-2 mb-2">
            <Star className="w-6 h-6 text-amber-500" />
            <span className="text-sm font-medium text-amber-500">VIP Program</span>
          </div>
          <h2 className="text-2xl font-bold mb-2">Join Elite Club</h2>
          <p className="text-muted-foreground mb-4">
            Exclusive rewards, personal manager, and premium benefits
          </p>
          <Button variant="outline" className="border-amber-500 text-amber-500 hover:bg-amber-500 hover:text-black">
            Learn More
          </Button>
        </div>
        <div className="absolute -right-4 -top-4 w-32 h-32 bg-amber-500/20 rounded-full blur-xl" />
        <div className="absolute -right-8 -bottom-8 w-24 h-24 bg-amber-500/30 rounded-full blur-lg" />
      </div>
    </div>
  );
}