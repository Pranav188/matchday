import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Premier League Forecasts | Matchday",
  description:
    "Explore Premier League home, draw and away probabilities from a leakage-aware match outcome model.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
