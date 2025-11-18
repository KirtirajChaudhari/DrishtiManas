import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#0F8EC7",
          dark: "#0B6FA0",
          light: "#39A8DA"
        },
        accent: "#FCAA67"
      }
    }
  },
  plugins: []
};

export default config;
