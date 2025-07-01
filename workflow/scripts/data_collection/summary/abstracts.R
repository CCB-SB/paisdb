##################################################
# IMPORTS
##################################################
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(ComplexHeatmap))

##################################################
# MAIN
##################################################

#
##  Visualization and stats
#

# Logging
log = snakemake@log[['log']]

# Pubmed abstracts results 
df_or = read.csv(snakemake@input[['csv']])
write(sprintf("Abstracts retrieved = %s", nrow(df_or)), file = log, append=TRUE)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Publications per year
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get distribution per years
per_years <- df_or %>% group_by(publication_year) %>% count()
per_years <- per_years %>%
  mutate(perc = round(n/nrow(df_or)*100, 2)) %>%
  mutate(perc_text = paste(perc, "%")) %>%
  filter(publication_year >= 1970)
per_years$publication_year <-  factor(per_years$publication_year)

hits_per_years <- per_years %>%
  ggplot(aes(x = publication_year, y = n)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label=perc_text), vjust=-0.3)+
  theme_minimal()+
  theme(axis.text.x = element_text(angle=45))+
  ggtitle("Publications per year")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## CUMULATIVE Years
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cum_years <- df_or %>%
  ggplot(aes(publication_year)) + 
  stat_ecdf(geom = "step") +
  ggtitle("Acummulative publications per year") +
  theme_minimal()

#
## Disease-Pathogen queries
#

# Summarize info by Disease-pathogen term and the number of pubmed articles (Hits)
df = df_or %>% group_by(pathogen, disease) %>% count() %>% arrange(desc(n)) %>% rename(hits = n)
write(sprintf("Disease-Pathogen queries with at least 1 hit = %s", nrow(df)), file = log, append=TRUE)
write.table(head(df, 15),
  file=log, append=TRUE)

# Hits by groups
df = df %>% mutate(
  hits_group = case_when(
    hits == 1 ~ "[1]",
    hits == 2 ~ "[2]",
    hits == 3 ~ "[3]",
    hits == 4 ~ "[4]",
    hits == 5 ~ "[5]",
    hits >= 6 & hits < 10 ~ "[6,10)",
    hits >= 10 & hits < 20 ~ "[10,20)",
    hits >= 20 & hits < 50 ~ "[20,50)",
    hits >= 50 & hits < 100 ~ "[50,100)",
    hits >= 100 & hits < 200 ~ "[100,200)",
    hits >= 200  ~ "[>=200]"
  ))


# Hits distribution by groups
df_groups <- df %>% group_by(hits_group) %>% count() 
df_groups <- df_groups %>% 
  mutate(perc = round(n/nrow(df)*100, 2)) %>%
  mutate(perc_text = paste(perc, "%")) 
df_groups$hits_group <-  factor(df_groups$hits_group,
  levels = c("[1]", "[2]", "[3]", "[4]", "[5]",
  "[6,10)", "[10,20)", "[20,50)","[50,100)", 
  "[100,200)","[>=200]"))

# Add hit groups to publications_df
df$query_key <- paste(df$disease, df$pathogen, sep=',')
values <-  df %>% select(c(query_key, hits_group))
df_or <- inner_join(df_or, values, by=join_by(query_key))
df_or$hits_group <-  factor(df_or$hits_group,
  levels = c("[1]", "[2]", "[3]", "[4]", "[5]",
  "[6,10)", "[10,20)", "[20,50)","[50,100)", 
  "[100,200)","[>=200]"))

hits_group_counts <- df_or %>% group_by(hits_group) %>% count()
hits_group_counts <- hits_group_counts %>% 
  mutate(perc = round(n/nrow(df_or)*100, 2)) %>%
  mutate(perc_text = paste(perc, "%")) 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Publications per hits_group
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
publications_per_groups <- ggplot(hits_group_counts, aes(x = hits_group, y = n)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label=perc_text), vjust=-0.3)+
  ggtitle("Publications per Disease-Pathogen group") +
  theme_minimal()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Disease-Pathogen query hits distribution
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hits_by_groups <- ggplot(df_groups, aes(x =hits_group, y = n)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label=perc_text), vjust=-0.3)+
  ggtitle("Disease-Pathogen query hits distribution (by groups)") +
  theme_minimal()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Disease-Pathogen query with ( 1 > hit < 50)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Filter queries with only 1 single hit
df1 = df %>% filter(hits == 1)
df_filt <- df %>% filter(hits > 1)
write(sprintf("Disease-Pathogen queries with 1 single hit: %s (%s percent.)",
  nrow(df1), (nrow(df1)/ nrow(df))*100 ), 
  file = log, append=TRUE)
write(sprintf("Filtering dataframe (> 1 hit): %s",
  nrow(df_filt)), 
  file = log, append=TRUE)

# Hit distribution up to 100
hits_upto100 <- df_filt %>% ggplot(aes(x = hits)) +
  geom_histogram(aes(y=(..count..)/sum(..count..)*100), bins=20) +
  xlim(1, 50) +
  ylab("Percentage (%)")+
  ggtitle("Hits per Disease-Pathogen query ( 1 > hit < 50)") +
  theme_minimal()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## CUMULATIVE hits per Disease-Pathogen query
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cum <- df_filt %>% 
  filter(hits <= 50) %>%
  ggplot(aes(hits)) + 
  stat_ecdf(geom = "step") +
  ggtitle("Acummulative hits per Disease-Pathogen query ( 1 > hit < 50)") +
  theme_minimal()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Publications from single disease-pathogen query hit per years
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get unique publication from query_terms with only 1 hit
terms_one_hit = paste(df1$disease, df1$pathogen, sep=",")
df_one = df_or %>% filter(query_key %in% terms_one_hit)

# Get distribution per years
per_years <- df_one %>% group_by(publication_year) %>% count()
per_years <- per_years %>%
  mutate(perc = round(n/nrow(df_one)*100, 2)) %>%
  mutate(perc_text = paste(perc, "%"))
per_years$publication_year <-  factor(per_years$publication_year)

hits_per_years_unique <- 
  per_years %>% 
  ggplot(aes(x = publication_year, y = n)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label=perc_text), vjust=-0.3)+
  theme_minimal()+
  theme(axis.text.x = element_text(angle=45))+
  ggtitle("Publications from single disease-pathogen hit per years")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Publications from 6-10 disease-pathogen query hits per years
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get unique publication from query_terms with 6-10 hits
df6 = df %>% filter(hits_group == '[6,10)')
terms = paste(df6$disease, df6$pathogen, sep=",")
d = df_or %>% filter(query_key %in% terms)

# Get distribution per years
per_years <- d %>% group_by(publication_year) %>% count()
per_years <- per_years %>%
  mutate(perc = round(n/nrow(d)*100, 2)) %>%
  mutate(perc_text = paste(perc, "%"))
per_years$publication_year <-  factor(per_years$publication_year)

hits_per_years_6 <- 
  per_years %>% 
  ggplot(aes(x = publication_year, y = n)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label=perc_text), vjust=-0.3)+
  theme_minimal()+
  theme(axis.text.x = element_text(angle=45))+
  ggtitle("Publications from 6-10 disease-pathogen hit per years")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## HEATMAP
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # Reshape the data for the heatmap
# ht_data <- dcast(df_filt, pathogen ~ disease, value.var = "hits", fun.aggregate = sum)
# ht_data <- ht_data %>% column_to_rownames(var = "pathogen") %>% as.matrix()

# top <- data.frame(rowSums(ht_data))
# colnames(top) <- "abundance"
# top_25 <- top %>% slice_max(abundance, n=25) %>% rownames()


# hp <- Heatmap(ht_data[top_25, ],
#   column_title = "Top25 abundant pathogens")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## NETWORK
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # Create a graph for the network 
# df_filt2 <- df_filt %>% filter(pathogen %in% top_25)                                               
# g <- graph_from_data_frame(df_filt2, directed = FALSE)

# # Calculate edge widths proportional to hits
# # Adjust the scale factor as needed
# scale_factor <- 10
# edge_widths <- E(g)$Hits / max(E(g)$Hits) * scale_factor

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## PATHOGENS HITS BY DISEASE (PARKINSON'S)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # Assuming df is your dataframe with columns 'Pathogen', 'Disease', 'Hits'
# # Filter for Parkinson's disease
# parkinsons_df <- subset(df_filt, disease == "Parkinson's disease")

# # Sort by hits in decreasing order
# parkinsons_df <- parkinsons_df[order(-parkinsons_df$hits),]

# # Create a bar plot
# bp_plot <- ggplot(parkinsons_df, aes(x = reorder(pathogen, hits), y = hits, fill = hits)) +
#   geom_bar(stat = "identity") +
#   scale_fill_gradient(low = "lightblue", high = "blue") +
#   coord_flip() +  # Flips the axes to make horizontal bars
#   labs(title = "Pathogens Linked to Parkinson's Disease by Hits",
#        x = "Pathogen",
#        y = "Number of Hits") +
#   theme_minimal()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## SAVE TO PDF
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pdf_out <- snakemake@output[['pdf']]
pdf(pdf_out, width=20, height=10)
print(hits_per_years)
print(cum_years)
print(publications_per_groups)
print(hits_by_groups)
print(hits_upto100)
print(cum)
print(hits_per_years_unique)
print(hits_per_years_6)
# print(hg_plot2)
# draw(hp)
# # Plot the graph with adjusted edge widths, without edge labelsg_plot
# plot(g, edge.width = edge_widths, vertex.color="lightblue", vertex.size=10, 
#      vertex.label.color="black", vertex.label.cex=0.8, edge.color="grey",
#      main="Network Graph of Pathogen-Disease Relationships")
# print(bp_plot)
dev.off()

