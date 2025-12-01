import { 
  Home, 
  Gamepad2, 
  Trophy, 
  Users, 
  CreditCard, 
  Settings, 
  Gift,
  Star,
  TrendingUp
} from "lucide-react";
import { Button } from "./ui/button";

export default function NavigationSidebar() {
  const menuItems = [
    { icon: Home, label: "Home", active: true },
    { icon: Gamepad2, label: "All Games" },
    { icon: Star, label: "Favorites" },
    { icon: TrendingUp, label: "Live Casino" },
    { icon: Trophy, label: "Tournaments" },
    { icon: Gift, label: "Promotions" },
    { icon: Users, label: "Community" },
    { icon: CreditCard, label: "Banking" },
    { icon: Settings, label: "Settings" },
  ];

  return (
    <div className="w-64 bg-card border-r border-border h-full flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <Gamepad2 className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="text-xl font-semibold">CasinoMax</span>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 p-4">
        <div className="space-y-2">
          {menuItems.map((item, index) => {
            const Icon = item.icon;
            return (
              <Button
                key={index}
                variant={item.active ? "default" : "ghost"}
                className={`w-full justify-start gap-3 h-12 ${
                  item.active 
                    ? "bg-primary text-primary-foreground hover:bg-primary/90" 
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"
                }`}
              >
                <Icon className="w-5 h-5" />
                {item.label}
              </Button>
            );
          })}
        </div>
      </nav>

      {/* User Profile */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center">
            <span className="text-primary-foreground font-medium">JD</span>
          </div>
          <div className="flex-1">
            <p className="font-medium">John Doe</p>
            <p className="text-sm text-muted-foreground">$1,234.56</p>
          </div>
        </div>
      </div>
    </div>
  );
}