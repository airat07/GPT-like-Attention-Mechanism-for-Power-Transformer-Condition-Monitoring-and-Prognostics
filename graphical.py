import tkinter as tk
# from tkinter import tkk
from PIL import ImageTk, Image 
from tkinter import filedialog as fd
# from tkinter import *

import datetime

import colorama
colorama.init()
from colorama import Fore, Style, Back
from simple_chat import chat


# Model: Simple Echo Chatbot
class SimpleChatbot():
    def get_response(self, message):
        tup = chat(message)
        return f"Bot: {tup[0]}", tup[1]
    
class window(tk.Tk):
    def __init__(self):
        super().__init__()
        #initiating the window
        self.title = ("Chat Bot")
        self.geometry('1160x582')
        self.resizable(True,True)
        self.create_widgets()
        self.return_Key()
        self.chatbot = SimpleChatbot()
        self.isgraph = False
        
    def create_widgets(self):
        #display pull-down and pop-up menu widget
        self.main_menu = tk.Menu(self)
        self.main_menu.add_command(label= 'Clear', command = self.clearScreen)
        #FOR FUTURE: Add load history command
        # the left chunk for logo
        self.pictureWindow = tk.Text(self, bd = 1, bg = "gray", width = "50", height = "8", font = ("Times New Roman", 30), foreground = "#00ffff", padx = 12, pady = 6)
        self.pictureWindow.place(x = 6, y = 6, height = 570, width = 375)
        self.pictureWindow.insert(tk.INSERT, "FPL CHATBOT")
        # Upload Image
        img = Image.open('FPL_Datasets/assets/fpl.jpg')
        img_new = img.resize((200, 230))  # Resize the image 
        self.photo = ImageTk.PhotoImage(img_new)
        # Create a frame to display the image
        self.logo = tk.Frame(self, width=200, height= 230)
        self.logo.place(x=90, y=330)
        #Input image into Frame
        self.label = tk.Label(self.logo,image=self.photo)
        self.label.pack(padx=5,pady=5)
        #displays all chats of current session
        self.chatWindow = tk.Text(self, bd = 1, bg = "black", blockcursor=True, width = "50", height = "8", font = ("Times New Roman", 14), foreground = "#00ffff")
        self.chatWindow.place(x = 387, y = 6, height = 450, width = 750)
        #placing the scrollbar
        self.scrollbar = tk.Scrollbar(self, command=self.chatWindow.yview, cursor="mouse")
        self.scrollbar.place(x = 1137, y = 6, height = 450)
        
        #All chat messages window
        self.messageWindow = tk.Text(self, bd=0, bg="black", width="30", height="4", font=("Times New Roman", 18), foreground="#ffffff")
        self.messageWindow.place(x = 387, y = 462, height = 115, width = 562)
        
        # the button to send
        self.sendButton = tk.Button(self, text="Send",  width="12", height=5,bd=0, bg="#0080ff", activebackground="#00bfff",foreground='black',font=("Arial", 16), command = self.send_message)
        self.sendButton.place(x= 955, y=462, height= 55, width = 195)
    
        #Button to clear screen
        self.clearButton= tk.Button(self, text="Clear Screen",  width="12", height=5,bd=0, bg="#0080ff", activebackground="#00bfff",foreground='black',font=("Arial", 16), command = self.clearScreen)
        self.clearButton.place(x= 955, y=522, height= 55, width = 195)

        #chat history file
        self.history_file = open("chat_history.txt", "a")

    def display_Graph(self,type):
        path = "/home/isense/Transformer/DataPlots/"
        path = path + type + "_plot.png"
        img = Image.open(path)
        img_resize = img.resize((352, 264))
        self.test = ImageTk.PhotoImage(img_resize)
        self.chatWindow.tag_configure("bot", foreground="green")
        self.chatWindow.insert(tk.END, "\n\nSee plot of your data entry on side: \n", "bot")
        self.graph = tk.Frame(self, width=352, height=264)
        self.graph.place(x=16, y=40)
        self.label = tk.Label(self.graph, image=self.test)
        self.label.pack()
        if self.graph:
            self.isgraph = True
        
        
    def send_message(self):
        if self.isgraph == True:
            self.graph.destroy()
        user_message = self.messageWindow.get("1.0", "end").strip()
        if user_message:
            self.display_message(f"User: {user_message}") 
            response, type = self.chatbot.get_response(user_message)
            self.display_message(response, bot=True)
            if type is not None:
                self.display_Graph(type)

            # Save chat history with timestamp header to the file
            time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.history_file.write(f"\n----- Session: {time_now} -----\n")
            self.history_file.write(f"User: {user_message}\n")
            self.history_file.write(f"Bot: {response}\n")
            self.history_file.flush()  # Flush to ensure data is written immediately

            self.messageWindow.delete("1.0", "end")
            
        else:
            pass
            # messagebox.showinfo("Error", "Please enter a message.")

    def display_message(self, message, bot=False):
        if bot:
            self.chatWindow.tag_configure("bot", foreground="green")
            message = "\n" + message
            self.chatWindow.insert(tk.END, message, "bot")
        else:
            self.chatWindow.tag_configure("user", foreground="blue")
            self.chatWindow.insert(tk.END, "\n" + message, "user")

    def __del__(self):
        # Close the chat history file when the GUI is closed
        self.history_file.close()
            
    def clearScreen(self):
        self.chatWindow.delete('1.0', 'end')
        self.graph.destroy()

    def return_Key(self):
        self.bind('<Return>', self.send_message_with_enter)

    def send_message_with_enter(self,event):
        self.send_message()

    def start(self):
        self.mainloop()




if __name__ == "__main__":
    window().start()



