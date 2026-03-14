import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title:       "NBA Shadow Trader",
  description: "Real-time quant trading dashboard for NBA prediction markets",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
